#!/usr/bin/env python3
"""Evaluate synthetic datasets with the same variance/graph scoring logic."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SPADUPA_DISABLE_AUTORUN", "1")


@dataclass
class SyntheticRecord:
    header: str
    sequence: str
    family: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate synthetic dataset")
    parser.add_argument("--fasta", required=True, help="Synthetic FASTA path")
    parser.add_argument("--labels", required=True, help="Labels JSON path")
    parser.add_argument("--output", default="output/synthetic_eval", help="Output folder")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=16, help="ESM batch size")
    parser.add_argument("--embed-cache", default=None, help="Embedding cache npz")
    parser.add_argument("--amp", action="store_true", help="Use AMP for ESM inference")
    parser.add_argument("--data-parallel", action="store_true", help="Use DataParallel if available")
    parser.add_argument("--mock-embeddings", action="store_true", help="Use random embeddings instead of ESM")
    parser.add_argument("--mock-dim", type=int, default=256, help="Embedding dim for mock mode")
    parser.add_argument("--alpha", type=float, default=0.6, help="Propagation alpha")
    parser.add_argument("--hops", type=int, default=2, help="Propagation hops")
    parser.add_argument("--top-k", type=int, default=30, help="Top-k for recall metric")
    return parser.parse_args()


def family_from_header(header: str) -> str:
    if "|" in header:
        return header.split("|", 1)[0]
    return "OTHER"


def load_records(fasta_path: str) -> List[SyntheticRecord]:
    from Bio import SeqIO

    records: List[SyntheticRecord] = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        family = family_from_header(rec.id)
        records.append(SyntheticRecord(header=rec.id, sequence=str(rec.seq).strip(), family=family))
    return records


def load_labels(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def align_reference_to_query(reference: str, query: str) -> Dict[int, Optional[int]]:
    from Bio.Align import PairwiseAligner

    if reference == query:
        return {i: i for i in range(len(reference))}

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1
    alignment = aligner.align(reference, query)[0]

    mapping: Dict[int, Optional[int]] = {i: None for i in range(len(reference))}
    ref_blocks, query_blocks = alignment.aligned
    for (r0, r1), (q0, q1) in zip(ref_blocks, query_blocks):
        for r_i, q_i in zip(range(r0, r1), range(q0, q1)):
            mapping[r_i] = q_i
    return mapping


def build_chain_graph(n_residues: int) -> np.ndarray:
    adj = np.zeros((n_residues, n_residues), dtype=np.float32)
    for i in range(n_residues):
        start = max(0, i - 5)
        stop = min(n_residues, i + 6)
        for j in range(start, stop):
            if i != j:
                adj[i, j] = 1.0
    adj += np.eye(n_residues, dtype=np.float32)
    degree = adj.sum(axis=1, keepdims=True)
    return adj / (degree + 1e-8)


def propagate_scores(initial: np.ndarray, adj_norm: np.ndarray, alpha: float, hops: int) -> np.ndarray:
    propagated = initial.copy()
    for _ in range(hops):
        propagated = alpha * initial + (1.0 - alpha) * (adj_norm @ propagated)
    return (propagated - propagated.min()) / (propagated.max() - propagated.min() + 1e-8)


def autocast_context(use_amp: bool, device: str):
    if not use_amp or device != "cuda":
        return nullcontext()
    try:
        import torch

        return torch.amp.autocast("cuda")
    except Exception:
        import torch

        return torch.cuda.amp.autocast()


def embed_sequences(
    records: Sequence[SyntheticRecord],
    device: str,
    batch_size: int,
    cache_path: Optional[str],
    use_amp: bool,
    use_data_parallel: bool,
) -> Dict[str, np.ndarray]:
    import esm
    import torch

    cache: Dict[str, np.ndarray] = {}
    if cache_path and Path(cache_path).exists():
        data = np.load(cache_path, allow_pickle=True)
        cache = {k: data[k] for k in data.files}

    missing = [rec for rec in records if rec.header not in cache]
    if not missing:
        return cache

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    if use_data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    batch_converter = alphabet.get_batch_converter()
    for start in range(0, len(missing), batch_size):
        chunk = missing[start : start + batch_size]
        batch = [(rec.header, rec.sequence) for rec in chunk]
        _, _, toks = batch_converter(batch)
        toks = toks.to(device)
        with torch.no_grad():
            with autocast_context(use_amp, device):
                out = model(toks, repr_layers=[33], return_contacts=False)
        reps = out["representations"][33]
        for i, rec in enumerate(chunk):
            cache[rec.header] = reps[i, 1 : len(rec.sequence) + 1].detach().cpu().numpy()

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **cache)

    return cache


def evaluate_family(
    family: str,
    records: Sequence[SyntheticRecord],
    embeddings: Dict[str, np.ndarray],
    hotspots: List[int],
    alpha: float,
    hops: int,
    top_k: int,
) -> Dict[str, float]:
    fam_records = [r for r in records if r.family == family]
    if not fam_records:
        return {
            "family": family,
            "n_sequences": 0,
            "roc_auc_variance": float("nan"),
            "roc_auc_graph": float("nan"),
            "roc_auc_combined": float("nan"),
            "recall_at_k_combined": float("nan"),
        }

    reference = fam_records[0]
    ref_seq = reference.sequence
    n_residues = len(ref_seq)

    per_position_vectors: List[List[np.ndarray]] = [[] for _ in range(n_residues)]
    for rec in fam_records:
        mapping = align_reference_to_query(ref_seq, rec.sequence)
        emb = embeddings[rec.header]
        for ref_idx, query_idx in mapping.items():
            if query_idx is not None and 0 <= query_idx < len(emb):
                per_position_vectors[ref_idx].append(emb[query_idx])

    variance = np.zeros(n_residues, dtype=np.float32)
    for idx, vectors in enumerate(per_position_vectors):
        if len(vectors) >= 2:
            stack = np.stack(vectors, axis=0)
            variance[idx] = np.var(stack, axis=0).mean()

    var_norm = (variance - variance.min()) / (variance.max() - variance.min() + 1e-8)
    propagated = propagate_scores(var_norm, build_chain_graph(n_residues), alpha=alpha, hops=hops)

    mid_positions = np.array([max(0, n_residues // 3), max(0, n_residues // 2)], dtype=int)
    dist = np.array([min(abs(i - a) for a in mid_positions) for i in range(n_residues)], dtype=np.float32)
    biophysical = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
    combined = 0.80 * propagated + 0.20 * biophysical

    y_true = np.array([1 if i in hotspots else 0 for i in range(n_residues)], dtype=int)

    from sklearn.metrics import roc_auc_score

    metrics = {
        "family": family,
        "n_sequences": len(fam_records),
        "n_hotspots": len(hotspots),
        "roc_auc_variance": float(roc_auc_score(y_true, var_norm)) if 0 < y_true.sum() < len(y_true) else float("nan"),
        "roc_auc_graph": float(roc_auc_score(y_true, propagated)) if 0 < y_true.sum() < len(y_true) else float("nan"),
        "roc_auc_combined": float(roc_auc_score(y_true, combined)) if 0 < y_true.sum() < len(y_true) else float("nan"),
    }

    top_idx = np.argsort(combined)[::-1][:top_k]
    metrics["recall_at_k_combined"] = float(len(set(top_idx) & set(hotspots)) / max(1, len(hotspots)))
    return metrics


def evaluate_dataset(
    fasta_path: str,
    labels_path: str,
    output_dir: Path,
    device: str,
    batch_size: int,
    embed_cache: Optional[str],
    use_amp: bool,
    use_data_parallel: bool,
    mock_embeddings: bool,
    mock_dim: int,
    alpha: float,
    hops: int,
    top_k: int,
) -> pd.DataFrame:
    records = load_records(fasta_path)
    labels = load_labels(labels_path)
    families = list(labels.get("families", {}).keys())

    if not records:
        raise SystemExit("No sequences found in synthetic FASTA")

    if mock_embeddings:
        rng = np.random.default_rng(42)
        embeddings = {
            rec.header: rng.normal(0, 1, size=(len(rec.sequence), mock_dim)).astype(np.float32)
            for rec in records
        }
    else:
        embeddings = embed_sequences(records, device, batch_size, embed_cache, use_amp, use_data_parallel)

    rows = []
    for fam in families:
        hotspots = labels["families"][fam]["hotspot_positions_0idx"]
        rows.append(evaluate_family(fam, records, embeddings, hotspots, alpha, hops, top_k))

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "synthetic_summary.csv", index=False)
    return df


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)

    device = args.device
    if args.mock_embeddings:
        device = "cpu"
    else:
        try:
            import torch

            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"

    t0 = time.time()
    df = evaluate_dataset(
        fasta_path=args.fasta,
        labels_path=args.labels,
        output_dir=output_dir,
        device=device,
        batch_size=max(1, args.batch_size),
        embed_cache=args.embed_cache,
        use_amp=args.amp,
        use_data_parallel=args.data_parallel,
        mock_embeddings=args.mock_embeddings,
        mock_dim=args.mock_dim,
        alpha=args.alpha,
        hops=args.hops,
        top_k=args.top_k,
    )
    if args.mock_embeddings:
        print("Synthetic evaluation ran in mock-embedding mode.")
    print(df.to_string(index=False))
    print(f"Saved {output_dir / 'synthetic_summary.csv'}")
    print(f"Elapsed: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
