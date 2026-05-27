#!/usr/bin/env python3
"""Real Kaggle runner for NDM/VIM/IMP family analysis with ESM-2.

This script:
- loads real ESM-2 embeddings via fair-esm (esm2_t33_650M_UR50D)
- uses the curated NDM-1 variant set from `_run_pipeline.py`
- uses VIM/IMP homologs from `data/b1_filtered.fasta`
- computes 2-hop alpha=0.6 graph propagation
- reports ROC-AUC for each enzyme/family

For VIM/IMP, ROC-AUC is measured against observed variant positions relative to
that family's canonical reference sequence in the FASTA.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path when running from scripts/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Avoid auto-running the full pipeline when importing _run_pipeline in Kaggle.
os.environ.setdefault("SPADUPA_DISABLE_AUTORUN", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real ESM-2 family runner")
    parser.add_argument(
        "--input",
        type=str,
        default="data/b1_filtered.fasta",
        help="FASTA with VIM/IMP homologs (default: data/b1_filtered.fasta)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/kaggle_real",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--family-order",
        type=str,
        default="NDM,VIM,IMP",
        help="Comma-separated family order; earlier families are processed first",
    )
    parser.add_argument(
        "--max-family-seqs",
        type=int,
        default=16,
        help="Maximum sequences per family to embed (excluding NDM curated variants)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Propagation alpha (default: 0.6)",
    )
    parser.add_argument(
        "--hops",
        type=int,
        default=2,
        help="Propagation iterations / hops (default: 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="ESM-2 batch size (default: 2)",
    )
    parser.add_argument(
        "--curated-mutations",
        type=str,
        default="data/curated_mutations.json",
        help=(
            "Optional JSON file of curated validated mutation positions per family "
            "(default: data/curated_mutations.json)."
        ),
    )
    parser.add_argument(
        "--curated-zero-indexed",
        action="store_true",
        help="Treat curated positions as 0-indexed (default assumes 1-indexed).",
    )
    return parser.parse_args()


NDM_FAMILY = "NDM"
VIM_FAMILY = "VIM"
IMP_FAMILY = "IMP"


def family_from_header(header: str) -> Optional[str]:
    text = header.upper()
    if re.search(r"BLA?NDM|NDM-?\d+|BLAN", text):
        return NDM_FAMILY
    if re.search(r"VIM-?\d+|BLBV", text):
        return VIM_FAMILY
    if re.search(r"IMP-?\d+|BLBI|BLA-?IMP", text):
        return IMP_FAMILY
    return None


@dataclass
class FamilyRecord:
    name: str
    header: str
    sequence: str


def load_family_records(fasta_path: str) -> Dict[str, List[FamilyRecord]]:
    from Bio import SeqIO

    grouped: Dict[str, List[FamilyRecord]] = defaultdict(list)
    for rec in SeqIO.parse(fasta_path, "fasta"):
        fam = family_from_header(rec.description)
        if fam is None:
            continue
        grouped[fam].append(FamilyRecord(name=rec.id, header=rec.description, sequence=str(rec.seq).strip()))
    return grouped


def choose_reference(records: Sequence[FamilyRecord], family: str) -> FamilyRecord:
    for rec in records:
        if family == NDM_FAMILY and re.search(r"NDM-1\b|blaNDM-1", rec.header, re.I):
            return rec
        if family == VIM_FAMILY and re.search(r"VIM-1\b", rec.header, re.I):
            return rec
        if family == IMP_FAMILY and re.search(r"IMP-1\b", rec.header, re.I):
            return rec
    return records[0]


def limit_records(records: Sequence[FamilyRecord], limit: int) -> List[FamilyRecord]:
    if len(records) <= limit:
        return list(records)
    ordered = list(records)
    ref = choose_reference(ordered, family_from_header(ordered[0].header) or "")
    kept = [ref]
    for rec in ordered:
        if rec.header == ref.header:
            continue
        kept.append(rec)
        if len(kept) >= limit:
            break
    return kept


def make_sequence_reference_data() -> Tuple[str, List[int]]:
    from _run_pipeline import KNOWN_NDM1_MUTATIONS, NDM1_SEQUENCE

    known_positions = sorted({v["position"] for v in KNOWN_NDM1_MUTATIONS.values()})
    return NDM1_SEQUENCE, known_positions


def leave_one_out_validation(scores: np.ndarray, known_positions: Sequence[int]) -> Dict[str, float]:
    """Leave-one-out validation based on rank of each known position."""
    if not known_positions:
        return {
            "loo_mean_rank": np.nan,
            "loo_median_rank": np.nan,
            "loo_mean_percentile": np.nan,
            "loo_fraction_top30": np.nan,
        }
    n_positions = len(scores)
    loo_ranks = []
    for held_out_pos in known_positions:
        if 0 <= held_out_pos < n_positions:
            held_out_score = scores[held_out_pos]
            rank = int(np.sum(scores > held_out_score) + 1)
            loo_ranks.append(rank)
    if not loo_ranks:
        return {
            "loo_mean_rank": np.nan,
            "loo_median_rank": np.nan,
            "loo_mean_percentile": np.nan,
            "loo_fraction_top30": np.nan,
        }
    return {
        "loo_mean_rank": float(np.mean(loo_ranks)),
        "loo_median_rank": float(np.median(loo_ranks)),
        "loo_mean_percentile": float(100 * (1 - np.mean(loo_ranks) / n_positions)),
        "loo_fraction_top30": float(sum(1 for r in loo_ranks if r <= 30) / len(loo_ranks)),
    }


def load_curated_mutations(path: str, zero_indexed: bool = False) -> Dict[str, List[int]]:
    """Load curated validated mutation positions from a JSON file."""
    if not path:
        return {}
    curated_path = Path(path)
    if not curated_path.exists():
        return {}
    with curated_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    curated: Dict[str, List[int]] = {}
    for family, value in data.items():
        positions: List[int] = []
        if isinstance(value, dict) and "positions" in value:
            positions = value.get("positions", [])
        elif isinstance(value, list):
            positions = value
        try:
            pos_list = [int(p) for p in positions]
        except (TypeError, ValueError):
            continue
        if not zero_indexed:
            pos_list = [p - 1 for p in pos_list if p > 0]
        curated[str(family).upper()] = [p for p in pos_list if p >= 0]
    return curated


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


def propagate_scores(initial: np.ndarray, adj_norm: np.ndarray, alpha: float = 0.6, hops: int = 2) -> np.ndarray:
    propagated = initial.copy()
    for _ in range(hops):
        propagated = alpha * initial + (1.0 - alpha) * (adj_norm @ propagated)
    propagated = (propagated - propagated.min()) / (propagated.max() - propagated.min() + 1e-8)
    return propagated


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


def project_variant_positions(reference: str, query: str) -> List[int]:
    mapping = align_reference_to_query(reference, query)
    variant_positions: List[int] = []
    for ref_idx, query_idx in mapping.items():
        if query_idx is None or ref_idx >= len(reference) or query_idx >= len(query):
            variant_positions.append(ref_idx)
        elif reference[ref_idx] != query[query_idx]:
            variant_positions.append(ref_idx)
    return sorted(set(variant_positions))


def project_embeddings_to_reference(
    reference: str,
    query: str,
    query_embedding: np.ndarray,
) -> Dict[int, np.ndarray]:
    mapping = align_reference_to_query(reference, query)
    projected: Dict[int, np.ndarray] = {}
    for ref_idx, query_idx in mapping.items():
        if query_idx is not None and 0 <= query_idx < len(query_embedding):
            projected[ref_idx] = query_embedding[query_idx]
    return projected


def get_observed_variant_positions(reference: FamilyRecord, records: Sequence[FamilyRecord]) -> List[int]:
    positions: set[int] = set()
    for rec in records:
        if rec.header == reference.header:
            continue
        positions.update(project_variant_positions(reference.sequence, rec.sequence))
    return sorted(positions)


def evaluate_label_set(scores: Dict[str, np.ndarray], positive_positions: Sequence[int]) -> Tuple[Dict[str, float], np.ndarray]:
    n_residues = len(scores["combined"])
    valid_positions = [p for p in positive_positions if 0 <= p < n_residues]
    y_true = np.array([1 if i in valid_positions else 0 for i in range(n_residues)], dtype=int)

    from sklearn.metrics import roc_auc_score

    metrics = {
        "n_positive_positions": int(y_true.sum()),
        "roc_auc_variance": np.nan,
        "roc_auc_graph": np.nan,
        "roc_auc_combined": np.nan,
    }
    if 0 < y_true.sum() < len(y_true):
        metrics["roc_auc_variance"] = float(roc_auc_score(y_true, scores["variance"]))
        metrics["roc_auc_graph"] = float(roc_auc_score(y_true, scores["graph"]))
        metrics["roc_auc_combined"] = float(roc_auc_score(y_true, scores["combined"]))

    loo = leave_one_out_validation(scores["combined"], valid_positions)
    metrics.update(loo)
    return metrics, y_true


def compute_family_scores(
    reference_record: FamilyRecord,
    records: Sequence[FamilyRecord],
    embeddings: Dict[str, np.ndarray],
    alpha: float,
    hops: int,
    family: str,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    ref_seq = reference_record.sequence
    n_residues = len(ref_seq)

    # Reference-position embeddings aggregated from homologs / variants.
    per_position_vectors: List[List[np.ndarray]] = [[] for _ in range(n_residues)]
    for rec in records:
        projected = project_embeddings_to_reference(ref_seq, rec.sequence, embeddings[rec.header])
        for pos, vec in projected.items():
            per_position_vectors[pos].append(vec)

    # Compute variance across homologs at each reference position.
    embedding_dim = next(iter(embeddings.values())).shape[1]
    if any(per_position_vectors):
        variance = np.zeros(n_residues, dtype=np.float32)
        for idx, vectors in enumerate(per_position_vectors):
            if len(vectors) >= 2:
                stack = np.stack(vectors, axis=0)
                variance[idx] = np.var(stack, axis=0).mean()
    else:
        variance = np.zeros(n_residues, dtype=np.float32)

    var_norm = (variance - variance.min()) / (variance.max() - variance.min() + 1e-8)
    adj_norm = build_chain_graph(n_residues)
    propagated = propagate_scores(var_norm, adj_norm, alpha=alpha, hops=hops)

    # Mild biophysical prior used in the pipeline: farther from active site is more tolerant.
    if family == NDM_FAMILY:
        active_positions = np.array([119, 121, 123, 188, 207, 249], dtype=int)
    else:
        active_positions = np.array([max(0, n_residues // 3), max(0, n_residues // 2)], dtype=int)
    dist = np.array([min(abs(i - a) for a in active_positions) for i in range(n_residues)], dtype=np.float32)
    biophysical = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
    combined = 0.80 * propagated + 0.20 * biophysical

    df = pd.DataFrame(
        {
            "position": np.arange(1, n_residues + 1),
            "residue": list(ref_seq),
            "variance": var_norm,
            "graph": propagated,
            "combined": combined,
            "biophysical": biophysical,
        }
    )
    scores = {
        "variance": var_norm,
        "graph": propagated,
        "combined": combined,
        "biophysical": biophysical,
    }
    return df, scores


def main() -> int:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    os.environ.setdefault("SPADUPA_SKIP_INSTALLS", "1")

    # Import runtime deps only when executing on Kaggle / GPU environment.
    import torch
    import esm

    from Bio import SeqIO

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    family_order = [x.strip().upper() for x in args.family_order.split(",") if x.strip()]

    # Load curated NDM data from the main pipeline.
    ndm_reference, ndm_known_positions = make_sequence_reference_data()
    from _run_pipeline import get_ndm_variants

    ndm_variants = get_ndm_variants()
    ndm_records = [FamilyRecord(name=name, header=name, sequence=seq) for name, seq in ndm_variants.items()]
    ndm_reference_record = next(r for r in ndm_records if r.name == "NDM-1")

    # Load VIM/IMP FASTA homologs.
    fasta_records = load_family_records(args.input)
    family_records: Dict[str, List[FamilyRecord]] = {
        NDM_FAMILY: ndm_records,
        VIM_FAMILY: limit_records(fasta_records.get(VIM_FAMILY, []), args.max_family_seqs),
        IMP_FAMILY: limit_records(fasta_records.get(IMP_FAMILY, []), args.max_family_seqs),
    }

    # Filter empty families and keep requested order.
    family_records = {k: v for k, v in family_records.items() if v}
    order = [fam for fam in family_order if fam in family_records]

    if not order:
        raise RuntimeError("No NDM/VIM/IMP sequences found for analysis.")

    # Real ESM-2 model.
    print("Loading real ESM-2 model: esm2_t33_650M_UR50D")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    print(f"ESM-2 ready on {device}")

    curated_map = load_curated_mutations(args.curated_mutations, args.curated_zero_indexed)

    summary_rows = []
    for family in order:
        records = family_records[family]
        reference = choose_reference(records, family)
        print(f"\n[{family}] reference: {reference.header}")
        print(f"[{family}] sequences: {len(records)}")

        # Embed all sequences with the real model.
        embeddings: Dict[str, np.ndarray] = {}
        names = [r.header for r in records]
        seqs = [r.sequence for r in records]
        batch_size = max(1, args.batch_size)
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            batch = [(r.header, r.sequence) for r in chunk]
            _, _, toks = batch_converter(batch)
            toks = toks.to(device)
            with torch.no_grad():
                out = model(toks, repr_layers=[33], return_contacts=False)
            for i, rec in enumerate(chunk):
                emb = out["representations"][33][i, 1 : len(rec.sequence) + 1].detach().cpu().numpy()
                embeddings[rec.header] = emb

        df, scores = compute_family_scores(
            reference, records, embeddings, alpha=args.alpha, hops=args.hops, family=family
        )

        if family == NDM_FAMILY:
            primary_positions = list(ndm_known_positions)
            label_type = "curated_resistance_positions"
        else:
            primary_positions = get_observed_variant_positions(reference, records)
            label_type = "observed_variant_positions"

        primary_metrics, primary_labels = evaluate_label_set(scores, primary_positions)
        primary_df = df.copy()
        primary_df["label"] = primary_labels

        out_dir = Path(args.output) / family.lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        primary_df.to_csv(out_dir / f"{family.lower()}_scores.csv", index=False)

        summary_rows.append(
            {
                "family": family,
                "reference": reference.header,
                "n_sequences": len(records),
                "label_type": label_type,
                "analysis_level": "primary",
                **primary_metrics,
            }
        )

        print(f"[{family}] positives: {primary_metrics['n_positive_positions']}")
        print(
            f"[{family}] ROC-AUC variance={primary_metrics['roc_auc_variance']}, graph={primary_metrics['roc_auc_graph']}, combined={primary_metrics['roc_auc_combined']}"
        )
        print(
            f"[{family}] LOOCV mean rank={primary_metrics['loo_mean_rank']}, top30={primary_metrics['loo_fraction_top30']}"
        )

        curated_positions = curated_map.get(family)
        if curated_positions:
            curated_metrics, curated_labels = evaluate_label_set(scores, curated_positions)
            curated_df = df.copy()
            curated_df["label"] = curated_labels
            curated_df.to_csv(out_dir / f"{family.lower()}_scores_high_conf.csv", index=False)

            summary_rows.append(
                {
                    "family": family,
                    "reference": reference.header,
                    "n_sequences": len(records),
                    "label_type": "curated_validated_mutations",
                    "analysis_level": "high_confidence",
                    **curated_metrics,
                }
            )
            print(
                f"[{family}] High-confidence ROC-AUC combined={curated_metrics['roc_auc_combined']}"
            )
        else:
            if args.curated_mutations:
                print(
                    f"[{family}] No curated mutations found in {args.curated_mutations}; skipping high-confidence analysis."
                )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(Path(args.output) / "enzyme_auc_summary.csv", index=False)
    print(f"\nWrote {Path(args.output) / 'enzyme_auc_summary.csv'}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
