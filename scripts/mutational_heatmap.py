#!/usr/bin/env python3
"""Generate an in silico mutational heatmap for a reference sequence.

For each position, iterate all single amino acid substitutions, compute the
ESM-2 embedding difference to the reference, run graph propagation, and record
scores in a CSV suitable for 3D mapping.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure repo root on sys.path and prevent auto-run.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("SPADUPA_DISABLE_AUTORUN", "1")

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB import PDBIO, PDBParser, is_aa
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.Align import PairwiseAligner


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="In silico mutational heatmap")
    parser.add_argument("--output", default="output/heatmap", help="Output folder")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=4, help="ESM batch size")
    parser.add_argument("--cache", default="output/heatmap/embeddings_cache.npz", help="Embedding cache path")
    parser.add_argument("--pdb-id", default="3SPU", help="PDB ID for visualization")
    parser.add_argument("--sequence-fasta", default=None, help="Optional FASTA with reference sequence")
    return parser.parse_args()


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


def embed_sequences(records: List[SeqRecord], device: str, batch_size: int, cache_path: str) -> Dict[str, np.ndarray]:
    import torch
    import esm

    cache: Dict[str, np.ndarray] = {}
    cache_file = Path(cache_path)
    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        cache = {k: data[k] for k in data.files}

    missing = [rec for rec in records if rec.id not in cache]
    if not missing:
        return cache

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    for start in range(0, len(missing), batch_size):
        chunk = missing[start : start + batch_size]
        batch = [(rec.id, str(rec.seq)) for rec in chunk]
        _, _, toks = batch_converter(batch)
        toks = toks.to(device)
        with torch.no_grad():
            out = model(toks, repr_layers=[33], return_contacts=False)
        for i, rec in enumerate(chunk):
            emb = out["representations"][33][i, 1 : len(rec.seq) + 1].detach().cpu().numpy()
            cache[rec.id] = emb

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, **cache)
    return cache


def download_pdb(pdb_id: str, output_path: Path) -> None:
    import urllib.request

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(output_path))


def write_bfactor_pdb(
    pdb_path: Path, output_path: Path, scores: Dict[int, float], ref_seq: str
) -> None:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("REF", str(pdb_path))
    chain = next(structure.get_chains())
    residues = [res for res in chain.get_residues() if is_aa(res, standard=True)]

    chain_seq = "".join(protein_letters_3to1.get(res.resname.title(), "X") for res in residues)

    # Align reference to chain sequence to map positions.
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1
    alignment = aligner.align(ref_seq, chain_seq)[0]

    mapping: Dict[int, int] = {}
    ref_blocks, chain_blocks = alignment.aligned
    for (r0, r1), (c0, c1) in zip(ref_blocks, chain_blocks):
        for r_i, c_i in zip(range(r0, r1), range(c0, c1)):
            mapping[r_i + 1] = c_i + 1  # 1-indexed

    for ref_pos, score in scores.items():
        chain_pos = mapping.get(ref_pos)
        if not chain_pos or chain_pos > len(residues):
            continue
        for atom in residues[chain_pos - 1]:
            atom.set_bfactor(float(score))

    io = PDBIO()
    io.set_structure(structure)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(output_path))


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    from _run_pipeline import NDM1_SEQUENCE, ACTIVE_SITE_RESIDUES

    if args.sequence_fasta:
        records = list(SeqIO.parse(args.sequence_fasta, "fasta"))
        if not records:
            raise SystemExit("No sequences found in FASTA.")
        reference = records[0]
    else:
        reference = SeqRecord(Seq(NDM1_SEQUENCE), id="NDM-1", description="NDM-1 reference")

    device = args.device
    try:
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    ref_seq = str(reference.seq)
    n_residues = len(ref_seq)
    adj_norm = build_chain_graph(n_residues)

    # Simple biophysical prior based on distance from active site residues.
    dist = np.array([min(abs(i - a) for a in ACTIVE_SITE_RESIDUES) for i in range(n_residues)], dtype=np.float32)
    biophysical = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)

    mutants: List[SeqRecord] = []
    for idx, ref_aa in enumerate(ref_seq, start=1):
        for mut_aa in AMINO_ACIDS:
            if mut_aa == ref_aa:
                continue
            mut_seq = list(ref_seq)
            mut_seq[idx - 1] = mut_aa
            mut_id = f"{ref_aa}{idx}{mut_aa}"
            mutants.append(SeqRecord(Seq("".join(mut_seq)), id=mut_id, description=mut_id))

    embeddings = embed_sequences([reference] + mutants, device=device, batch_size=args.batch_size, cache_path=args.cache)
    ref_emb = embeddings[reference.id]

    rows = []
    per_position_scores: Dict[int, List[float]] = {i: [] for i in range(1, n_residues + 1)}

    for mutant in mutants:
        mut_emb = embeddings[mutant.id]
        diff = mut_emb - ref_emb
        variance = (diff * diff).mean(axis=1) / 4.0
        var_norm = (variance - variance.min()) / (variance.max() - variance.min() + 1e-8)
        graph_scores = propagate_scores(var_norm, adj_norm)
        combined = 0.80 * graph_scores + 0.20 * biophysical

        ref_aa = mutant.id[0]
        position = int(mutant.id[1:-1])
        mut_aa = mutant.id[-1]
        score = float(combined[position - 1])
        rows.append(
            {
                "position": position,
                "ref_aa": ref_aa,
                "mut_aa": mut_aa,
                "variance": float(var_norm[position - 1]),
                "graph": float(graph_scores[position - 1]),
                "biophysical": float(biophysical[position - 1]),
                "combined": score,
            }
        )
        per_position_scores[position].append(score)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "mutational_heatmap.csv", index=False)

    summary_rows = []
    for pos in range(1, n_residues + 1):
        mean_score = float(np.mean(per_position_scores[pos])) if per_position_scores[pos] else float("nan")
        summary_rows.append(
            {
                "position": pos,
                "ref_aa": ref_seq[pos - 1],
                "mean_combined": mean_score,
                "active_site": 1 if pos - 1 in ACTIVE_SITE_RESIDUES else 0,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "mutational_heatmap_summary.csv", index=False)

    pdb_path = output_dir / f"{args.pdb_id}.pdb"
    if not pdb_path.exists():
        download_pdb(args.pdb_id, pdb_path)

    score_map = {row["position"]: row["mean_combined"] for row in summary_rows}
    write_bfactor_pdb(pdb_path, output_dir / "mutational_heatmap_mean.pdb", score_map, ref_seq)

    print(f"Saved heatmap: {output_dir / 'mutational_heatmap.csv'}")
    print(f"Saved summary: {output_dir / 'mutational_heatmap_summary.csv'}")
    print(f"Saved PDB: {output_dir / 'mutational_heatmap_mean.pdb'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
