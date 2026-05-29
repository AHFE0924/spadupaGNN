#!/usr/bin/env python3
"""Compute residue-level importance from ESM-2 embeddings and map to structure.

Trains a simple logistic regression model on per-residue embeddings (NDM-1)
using curated resistance-associated positions as positives. Importance is
computed per residue and written to CSV and a PDB with B-factors set to
importance values.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Avoid auto-running the full pipeline when importing _run_pipeline.
os.environ.setdefault("SPADUPA_DISABLE_AUTORUN", "1")

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBIO, PDBParser, is_aa
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residue importance via embeddings")
    parser.add_argument("--output", default="output/importance", help="Output folder")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--cache", default="output/embeddings_cache.npz", help="Embedding cache path")
    parser.add_argument("--pdb-id", default="3SPU", help="PDB ID for structure mapping")
    parser.add_argument(
        "--importance-method",
        choices=["coef", "permutation"],
        default="coef",
        help="Residue importance method (default: coef)",
    )
    return parser.parse_args()


def embed_sequences(records: List[SeqIO.SeqRecord], device: str, cache_path: str) -> Dict[str, np.ndarray]:
    import torch
    import esm

    cache: Dict[str, np.ndarray] = {}
    if cache_path and Path(cache_path).exists():
        data = np.load(cache_path, allow_pickle=True)
        cache = {k: data[k] for k in data.files}

    missing = [rec for rec in records if rec.id not in cache]
    if not missing:
        return cache

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    for rec in missing:
        batch = [(rec.id, str(rec.seq))]
        _, _, toks = batch_converter(batch)
        toks = toks.to(device)
        with torch.no_grad():
            out = model(toks, repr_layers=[33], return_contacts=False)
        emb = out["representations"][33][0, 1 : len(rec.seq) + 1].detach().cpu().numpy()
        cache[rec.id] = emb

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **cache)

    return cache


def align_sequences(ref_seq: str, query_seq: str) -> Dict[int, Optional[int]]:
    if ref_seq == query_seq:
        return {i: i for i in range(len(ref_seq))}

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1
    alignment = aligner.align(ref_seq, query_seq)[0]

    mapping: Dict[int, Optional[int]] = {i: None for i in range(len(ref_seq))}
    ref_blocks, query_blocks = alignment.aligned
    for (r0, r1), (q0, q1) in zip(ref_blocks, query_blocks):
        for r_i, q_i in zip(range(r0, r1), range(q0, q1)):
            mapping[r_i] = q_i
    return mapping


def map_importance_to_pdb(
    pdb_path: Path,
    output_path: Path,
    ref_seq: str,
    importance: np.ndarray,
) -> None:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("NDM", str(pdb_path))

    chain = next(structure.get_chains())
    chain_residues = [res for res in chain.get_residues() if is_aa(res, standard=True)]
    from Bio.Data.IUPACData import protein_letters_3to1

    chain_seq = "".join(protein_letters_3to1.get(res.resname.title(), "X") for res in chain_residues)

    mapping = align_sequences(ref_seq, chain_seq)
    for ref_idx, chain_idx in mapping.items():
        if chain_idx is None or ref_idx >= len(importance):
            continue
        if chain_idx >= len(chain_residues):
            continue
        score = float(importance[ref_idx])
        for atom in chain_residues[chain_idx]:
            atom.set_bfactor(score)

    io = PDBIO()
    io.set_structure(structure)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(output_path))


def download_pdb(pdb_id: str, output_path: Path) -> None:
    import urllib.request

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(output_path))


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    from _run_pipeline import ACTIVE_SITE_RESIDUES, KNOWN_NDM1_MUTATIONS, NDM1_SEQUENCE, get_ndm_variants

    ndm_variants = get_ndm_variants()
    records = [SeqRecord(Seq(seq), id=name, description=name) for name, seq in ndm_variants.items()]

    device = args.device
    try:
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    embeddings = embed_sequences(records, device=device, cache_path=args.cache)

    # Use mean embedding per residue across variants
    emb_stack = np.stack([embeddings[r.id] for r in records], axis=0)
    mean_embedding = emb_stack.mean(axis=0)

    X = mean_embedding
    known_positions = sorted({v["position"] for v in KNOWN_NDM1_MUTATIONS.values()})
    y = np.array([1 if i in known_positions else 0 for i in range(X.shape[0])])

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs) if y.sum() > 0 else float("nan")

    if args.importance_method == "permutation":
        from sklearn.inspection import permutation_importance

        perm = permutation_importance(
            model, X, y, n_repeats=30, random_state=42, scoring="roc_auc"
        )
        dim_importance = np.abs(perm.importances_mean)
        importance = np.sum(np.abs(X) * dim_importance, axis=1)
    else:
        coef = model.coef_.reshape(-1)
        importance = np.sum(np.abs(X * coef), axis=1)
    if importance.max() > 0:
        importance = importance / importance.max()

    df = pd.DataFrame(
        {
            "position": np.arange(1, len(NDM1_SEQUENCE) + 1),
            "residue": list(NDM1_SEQUENCE),
            "label": y,
            "probability": probs,
            "importance": importance,
            "active_site": [1 if i in ACTIVE_SITE_RESIDUES else 0 for i in range(len(NDM1_SEQUENCE))],
        }
    )
    df.to_csv(output_dir / "ndm1_residue_importance.csv", index=False)

    pdb_path = output_dir / f"{args.pdb_id}.pdb"
    if not pdb_path.exists():
        download_pdb(args.pdb_id, pdb_path)

    map_importance_to_pdb(pdb_path, output_dir / "ndm1_residue_importance.pdb", NDM1_SEQUENCE, importance)

    print(f"AUC (embedding logistic regression): {auc:.4f}")
    print(f"Saved: {output_dir / 'ndm1_residue_importance.csv'}")
    print(f"Saved: {output_dir / 'ndm1_residue_importance.pdb'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
