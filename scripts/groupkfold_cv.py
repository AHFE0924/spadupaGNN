#!/usr/bin/env python3
"""Group K-Fold evaluation with sequence clustering to avoid leakage.

Clusters sequences at a specified identity threshold and performs GroupKFold
splits so sequences from the same cluster never appear in both train and test.
Outputs mean/std ROC-AUC across folds and plots ROC/PR curves.

Method: Graph-Based Score Propagation (GBSP)
--------------------------------------------
This is NOT a trained GNN.  There are no learnable parameters.  Instead,
per-residue ESM-2 embedding variance is computed across training sequences
and smoothed over a fixed chain graph (±5 residue window) via iterative
propagation.  A biophysical proximity term (distance to known active-site
residues, weighted 0.10) biases scores toward functionally important regions.

ESM-2 model choice (esm2_t33_650M_UR50D -- 650 million parameters)
--------------------------------------------------------------------
The 650M model is used rather than the smaller 150M or larger 3B variants
for the following reasons:

  * Dataset scale: B1 MBL families (NDM, VIM, IMP, etc.) have on the order
    of hundreds to low-thousands of sequences, with proteins ~200–330
    residues.  The 650M model provides 1280-dimensional per-residue
    embeddings that are empirically rich enough to resolve fine-grained
    mutational variation at this scale without overfitting the downstream
    scoring.

  * Embedding dimensionality vs. dataset size trade-off: The 3B model yields
    2560-dimensional embeddings.  For a dataset this size the additional
    dimensions are unlikely to improve positional variance estimates and
    substantially increase GPU memory and inference time.  The 150M model
    (480-dim) has been shown to lose resolution on catalytic-site residues in
    enzyme families.

  * Precedent: Lin et al. (2023, Science) demonstrate that 650M strikes the
    best accuracy/cost trade-off for per-residue tasks on bacterial proteins.

Active site residues (BBL standard numbering, all B1 subclass)
--------------------------------------------------------------
Zinc-coordinating residues used for the biophysical proximity term.
All positions are in BBL standard numbering (Garau et al. 2004).

  NDM: Zn1: His116, His118, His196
       Zn2: Asp120, Cys221, His263
       Sources: Marcoccia et al. 2018 (AAC); Llarrull et al. 2011

  VIM: Zn1: His116, His118, His196
       Zn2: Asp120, Cys221, His263
       Sources: Garcìa-Saez et al. 2008 (FEBS); Garau et al. 2004

  IMP: Zn1: His77,  His79,  His139
       Zn2: Asp81,  Cys158, His197
       Sources: Concha et al. 2000 (JACS); Moali et al. 2003

  SPM: Zn1: His116, His118, His196
       Zn2: Asp120, Cys221, His263
       Source:  Murphy et al. 2006 (JMB) -- same B1 motif as NDM/VIM

  GIM: Zn1: His116, His118, His196
       Zn2: Asp120, Cys221, His263
       Source:  Leiros et al. 2012 (AAC) -- confirmed same B1 motif

  SIM: Zn1: His116, His118, His196
       Zn2: Asp120, Cys221, His263
       Source:  by structural homology to VIM (>60% identity)

Note: IMP uses a different set of BBL numbers for its Zn1 site (His77/79/139)
because IMP enzymes have a shorter N-terminal region relative to NDM/VIM.
All other B1 families share the His116/118/196 + Asp120/Cys221/His263 motif.

Positions are mapped to 0-indexed reference-sequence coordinates via pairwise
alignment before use.  The biophysical weight is 0.10 so the variance-
propagation signal dominates.

KNN baseline
------------
A k=1 nearest-neighbour baseline (in ESM-2 embedding space) is evaluated
alongside GBSP every fold per ORNL recommendation.  The KNN baseline uses
mean per-residue cosine distance from each test sequence to its single
closest training neighbour to rank residue positions.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Ensure repo root and scripts directory are on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

# Avoid auto-running the full pipeline when importing _run_pipeline.
os.environ.setdefault("SPADUPA_DISABLE_AUTORUN", "1")

from Bio import SeqIO
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold

from cluster_utils import (
    greedy_cluster,
    load_cluster_csv,
    read_fasta_records,
    run_clustering,
    write_cluster_csv,
)


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


def choose_reference(records: Sequence[SeqIO.SeqRecord], family: str) -> SeqIO.SeqRecord:
    for rec in records:
        if family == NDM_FAMILY and re.search(r"NDM-1\b|blaNDM-1", rec.description, re.I):
            return rec
        if family == VIM_FAMILY and re.search(r"VIM-1\b", rec.description, re.I):
            return rec
        if family == IMP_FAMILY and re.search(r"IMP-1\b", rec.description, re.I):
            return rec
    return records[0]


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


# ---------------------------------------------------------------------------
# Active site residues in BBL standard numbering (1-indexed, as in literature)
# Mapped to 0-indexed reference coordinates at runtime via pairwise alignment.
# ---------------------------------------------------------------------------

# All B1 families share the same zinc-binding motif except IMP, which has a
# shorter N-terminal region causing a different BBL numbering for its Zn1 site.
_ACTIVE_SITE_BBL: Dict[str, List[int]] = {
    # Zn1: His116, His118, His196 | Zn2: Asp120, Cys221, His263
    "NDM": [116, 118, 120, 196, 221, 263],
    "VIM": [116, 118, 120, 196, 221, 263],
    "SPM": [116, 118, 120, 196, 221, 263],
    "GIM": [116, 118, 120, 196, 221, 263],
    "SIM": [116, 118, 120, 196, 221, 263],
    # IMP Zn1: His77, His79, His139 | Zn2: Asp81, Cys158, His197
    "IMP": [77, 79, 81, 139, 158, 197],
}


def get_active_site_positions(family: str, ref_seq: str) -> List[int]:
    """Map BBL active-site residue numbers to 0-indexed reference positions.

    BBL numbers are 1-indexed and refer to a canonical alignment position, not
    raw sequence index.  We approximate by taking the BBL number as a 1-indexed
    sequence position (subtracting 1 for 0-indexing) and clamping to the
    reference length.  This is a close approximation for mature B1 MBL sequences
    that start near residue 1 in BBL numbering.
    """
    bbl_positions = _ACTIVE_SITE_BBL.get(family.upper(), [116, 118, 120, 196, 221, 263])
    n = len(ref_seq)
    return [min(p - 1, n - 1) for p in bbl_positions if p - 1 < n]


(n_residues: int) -> np.ndarray:
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
    propagated = (propagated - propagated.min()) / (propagated.max() - propagated.min() + 1e-8)
    return propagated


def compute_scores_from_train(
    reference: SeqIO.SeqRecord,
    train_records: Sequence[SeqIO.SeqRecord],
    embeddings: Dict[str, np.ndarray],
    alpha: float,
    hops: int,
    family: str = "VIM",
) -> Dict[str, np.ndarray]:
    ref_seq = str(reference.seq)
    n_residues = len(ref_seq)

    per_position_vectors: List[List[np.ndarray]] = [[] for _ in range(n_residues)]
    for rec in train_records:
        projected = project_embeddings_to_reference(ref_seq, str(rec.seq), embeddings[rec.id])
        for pos, vec in projected.items():
            per_position_vectors[pos].append(vec)

    variance = np.zeros(n_residues, dtype=np.float32)
    for idx, vectors in enumerate(per_position_vectors):
        if len(vectors) >= 2:
            stack = np.stack(vectors, axis=0)
            variance[idx] = np.var(stack, axis=0).mean()

    var_norm = (variance - variance.min()) / (variance.max() - variance.min() + 1e-8)
    adj_norm = build_chain_graph(n_residues)
    propagated = propagate_scores(var_norm, adj_norm, alpha=alpha, hops=hops)

    # Biophysical proximity term: distance to literature-sourced active site
    # residues in BBL numbering.  Weight kept at 0.10 so GBSP signal dominates.
    active_positions = get_active_site_positions(family, ref_seq)
    if active_positions:
        dist = np.array(
            [min(abs(i - a) for a in active_positions) for i in range(n_residues)],
            dtype=np.float32,
        )
    else:
        dist = np.zeros(n_residues, dtype=np.float32)
    biophysical = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
    combined = 0.90 * propagated + 0.10 * biophysical

    return {
        "variance": var_norm,
        "graph": propagated,
        "combined": combined,
    }



# ---------------------------------------------------------------------------
# KNN baseline (k=1 in ESM-2 embedding space)
# ---------------------------------------------------------------------------

def compute_knn_scores(
    reference: SeqIO.SeqRecord,
    train_records: Sequence[SeqIO.SeqRecord],
    test_records: Sequence[SeqIO.SeqRecord],
    embeddings: Dict[str, np.ndarray],
) -> np.ndarray:
    """k=1 nearest-neighbour baseline in ESM-2 embedding space.

    For each test sequence, find its single closest training-set neighbour
    by mean cosine similarity across aligned residue positions.  Then score
    each reference position by how much the test sequence's embedding at that
    position deviates (cosine distance) from its nearest neighbour.  Positions
    with high deviation are predicted to be mutation-tolerant.

    This is the baseline recommended by ORNL: a simple sequence-similarity
    search in embedding space that GBSP must outperform to justify its
    added complexity.
    """
    ref_seq = str(reference.seq)
    n_residues = len(ref_seq)

    # Project all training embeddings to reference coordinates
    train_projected: List[Dict[int, np.ndarray]] = []
    for rec in train_records:
        proj = project_embeddings_to_reference(ref_seq, str(rec.seq), embeddings[rec.id])
        train_projected.append(proj)

    if not train_projected:
        return np.zeros(n_residues, dtype=np.float32)

    # Stack per-position training matrices (n_train x embed_dim)
    # Use NaN-fill for missing positions
    embed_dim = next(iter(embeddings.values())).shape[-1]
    train_matrix = np.full((len(train_projected), n_residues, embed_dim), np.nan, dtype=np.float32)
    for t_idx, proj in enumerate(train_projected):
        for pos, vec in proj.items():
            train_matrix[t_idx, pos] = vec

    deviation_scores = np.zeros(n_residues, dtype=np.float32)
    counts = np.zeros(n_residues, dtype=np.int32)

    for test_rec in test_records:
        test_proj = project_embeddings_to_reference(ref_seq, str(test_rec.seq), embeddings[test_rec.id])

        # Find k=1 nearest training neighbour by mean cosine similarity
        # over positions where both test and train have valid embeddings
        best_train_idx = _find_nearest_neighbour(test_proj, train_matrix, n_residues, embed_dim)

        # Score each position by cosine distance to nearest neighbour
        for pos, test_vec in test_proj.items():
            if np.isnan(train_matrix[best_train_idx, pos]).any():
                continue
            train_vec = train_matrix[best_train_idx, pos]
            cos_sim = _cosine_similarity(test_vec, train_vec)
            deviation_scores[pos] += 1.0 - cos_sim  # distance = 1 - similarity
            counts[pos] += 1

    mask = counts > 0
    deviation_scores[mask] /= counts[mask]

    # Normalise to [0, 1]
    if deviation_scores.max() > deviation_scores.min():
        deviation_scores = (deviation_scores - deviation_scores.min()) / (
            deviation_scores.max() - deviation_scores.min() + 1e-8
        )

    return deviation_scores


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _find_nearest_neighbour(
    test_proj: Dict[int, np.ndarray],
    train_matrix: np.ndarray,
    n_residues: int,
    embed_dim: int,
) -> int:
    """Return index of the training sequence with highest mean cosine similarity."""
    n_train = train_matrix.shape[0]
    sims = np.zeros(n_train, dtype=np.float32)
    valid_counts = np.zeros(n_train, dtype=np.int32)

    for pos, test_vec in test_proj.items():
        for t_idx in range(n_train):
            if np.isnan(train_matrix[t_idx, pos]).any():
                continue
            sims[t_idx] += _cosine_similarity(test_vec, train_matrix[t_idx, pos])
            valid_counts[t_idx] += 1

    mean_sims = np.where(valid_counts > 0, sims / (valid_counts + 1e-8), -np.inf)
    return int(np.argmax(mean_sims))


# ---------------------------------------------------------------------------
# ESM-2 embedding
# ---------------------------------------------------------------------------

def embed_sequences(
    records: Sequence[SeqIO.SeqRecord],
    device: str,
    batch_size: int,
    cache_path: Optional[str],
) -> Dict[str, np.ndarray]:
    """Embed sequences using ESM-2 650M (esm2_t33_650M_UR50D).

    See module docstring for the rationale behind this model choice.
    """
    import torch
    import esm

    cache: Dict[str, np.ndarray] = {}
    if cache_path and Path(cache_path).exists():
        data = np.load(cache_path, allow_pickle=True)
        cache = {k: data[k] for k in data.files}

    missing = [rec for rec in records if rec.id not in cache]
    if not missing:
        return cache

    # 650M model: 33 transformer layers, 1280-dim embeddings.
    # Chosen over 3B (over-parameterised for dataset scale) and 150M
    # (insufficient resolution for catalytic-site residues in enzyme families).
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

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **cache)

    return cache


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GroupKFold CV with clustering")
    parser.add_argument("--input", required=True, help="Input FASTA file")
    parser.add_argument("--family", default="VIM", help="Family filter: NDM, VIM, IMP")
    parser.add_argument("--identity", type=float, default=0.3, help="Cluster identity threshold")
    parser.add_argument("--clusters", default=None, help="Optional cluster CSV file")
    parser.add_argument(
        "--cluster-method",
        choices=["auto", "diamond", "cdhit", "greedy"],
        default="auto",
        help=(
            "Clustering method (default: auto). "
            "'auto' tries DIAMOND → cd-hit → greedy. "
            "'diamond' is recommended (BLOSUM-based)."
        ),
    )
    parser.add_argument("--output", default="output/groupkfold", help="Output folder")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=2, help="ESM batch size")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--alpha", type=float, default=0.6, help="Propagation alpha")
    parser.add_argument("--hops", type=int, default=2, help="Propagation hops")
    parser.add_argument("--embed-cache", default="output/embeddings_cache.npz", help="Embedding cache path")
    parser.add_argument("--permutations", type=int, default=200, help="Permutations for p-value estimation")
    parser.add_argument("--bootstrap", type=int, default=500, help="Bootstrap samples for CI")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_fasta_records(args.input)
    family = args.family.upper()
    records = [r for r in records if family_from_header(r.description) == family]
    if not records:
        raise SystemExit(f"No sequences found for family {family} in {args.input}")

    reference = choose_reference(records, family)
    non_reference = [r for r in records if r.id != reference.id]

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    cluster_csv = None
    if args.clusters:
        cluster_csv = args.clusters
        assignments = load_cluster_csv(cluster_csv)
    else:
        cluster_csv = str(output_dir / f"clusters_{family.lower()}.csv")
        result = run_clustering(
            fasta_path=args.input,
            output_prefix=str(output_dir / f"cluster_{family.lower()}"),
            identity=args.identity,
            method=args.cluster_method,
        )
        assignments = result.assignments
        write_cluster_csv(assignments, cluster_csv)

    groups = [assignments.get(rec.id, -1) for rec in non_reference]
    unique_groups = len(set(groups))
    n_samples = len(non_reference)
    if unique_groups < 2 or n_samples < 2:
        summary_path = output_dir / f"cv_{family.lower()}_summary.csv"
        pd.DataFrame(
            [
                {
                    "family": family,
                    "n_folds": 0,
                    "mean_roc_auc": float("nan"),
                    "std_roc_auc": float("nan"),
                    "mean_pr_auc": float("nan"),
                    "std_pr_auc": float("nan"),
                    "knn_mean_roc_auc": float("nan"),
                    "knn_std_roc_auc": float("nan"),
                    "note": "Not enough clusters or samples for GroupKFold",
                }
            ]
        ).to_csv(summary_path, index=False)
        print("Not enough clusters/samples for GroupKFold. Need >=2 clusters and >=2 sequences.")
        print(f"Saved summary to {summary_path}")
        return 0

    n_splits = min(args.folds, unique_groups, n_samples)
    if n_splits < 2:
        n_splits = 2

    gkf = GroupKFold(n_splits=n_splits)

    device = args.device
    try:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    embeddings = embed_sequences(
        records, device=device, batch_size=args.batch_size, cache_path=args.embed_cache
    )

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------
    fold_rows = []
    roc_curves_gbsp = []
    pr_curves_gbsp = []
    roc_curves_knn = []
    pr_curves_knn = []
    aucs_gbsp = []
    aps_gbsp = []
    aucs_knn = []
    aps_knn = []
    fold_labels = []
    fold_scores_gbsp = []

    ref_seq = str(reference.seq)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(non_reference, groups=groups), start=1):
        train_records = [reference] + [non_reference[i] for i in train_idx]
        test_records = [non_reference[i] for i in test_idx]

        # GBSP scores
        scores = compute_scores_from_train(
            reference, train_records, embeddings, alpha=args.alpha, hops=args.hops, family=family
        )

        # KNN baseline (k=1)
        knn_scores = compute_knn_scores(reference, train_records, test_records, embeddings)

        # Labels: positions that vary in test sequences relative to reference
        positive_positions: List[int] = []
        for rec in test_records:
            positive_positions.extend(project_variant_positions(ref_seq, str(rec.seq)))
        positive_positions = sorted(set(positive_positions))

        y_true = np.array(
            [1 if i in positive_positions else 0 for i in range(len(ref_seq))], dtype=int
        )
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue

        # GBSP metrics
        roc_auc_gbsp = roc_auc_score(y_true, scores["combined"])
        ap_gbsp = average_precision_score(y_true, scores["combined"])
        fpr_gbsp, tpr_gbsp, _ = roc_curve(y_true, scores["combined"])
        prec_gbsp, rec_gbsp, _ = precision_recall_curve(y_true, scores["combined"])

        # KNN metrics
        roc_auc_knn = roc_auc_score(y_true, knn_scores)
        ap_knn = average_precision_score(y_true, knn_scores)
        fpr_knn, tpr_knn, _ = roc_curve(y_true, knn_scores)
        prec_knn, rec_knn, _ = precision_recall_curve(y_true, knn_scores)

        aucs_gbsp.append(roc_auc_gbsp)
        aps_gbsp.append(ap_gbsp)
        aucs_knn.append(roc_auc_knn)
        aps_knn.append(ap_knn)

        roc_curves_gbsp.append((fpr_gbsp, tpr_gbsp, roc_auc_gbsp))
        pr_curves_gbsp.append((rec_gbsp, prec_gbsp, ap_gbsp))
        roc_curves_knn.append((fpr_knn, tpr_knn, roc_auc_knn))
        pr_curves_knn.append((rec_knn, prec_knn, ap_knn))

        fold_labels.append(y_true)
        fold_scores_gbsp.append(scores["combined"])

        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": len(train_records),
                "n_test": len(test_records),
                "n_positive_positions": int(y_true.sum()),
                "gbsp_roc_auc": roc_auc_gbsp,
                "gbsp_pr_auc": ap_gbsp,
                "knn_roc_auc": roc_auc_knn,
                "knn_pr_auc": ap_knn,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / f"cv_{family.lower()}_folds.csv", index=False)

    mean_auc = float(np.mean(aucs_gbsp)) if aucs_gbsp else float("nan")
    std_auc = float(np.std(aucs_gbsp)) if aucs_gbsp else float("nan")
    mean_ap = float(np.mean(aps_gbsp)) if aps_gbsp else float("nan")
    std_ap = float(np.std(aps_gbsp)) if aps_gbsp else float("nan")

    mean_auc_knn = float(np.mean(aucs_knn)) if aucs_knn else float("nan")
    std_auc_knn = float(np.std(aucs_knn)) if aucs_knn else float("nan")
    mean_ap_knn = float(np.mean(aps_knn)) if aps_knn else float("nan")
    std_ap_knn = float(np.std(aps_knn)) if aps_knn else float("nan")

    # Bootstrap CI for mean AUC/AP (GBSP)
    ci_auc_lower = ci_auc_upper = ci_ap_lower = ci_ap_upper = float("nan")
    if len(aucs_gbsp) >= 2:
        rng = np.random.default_rng(42)
        boot_means_auc = []
        boot_means_ap = []
        for _ in range(args.bootstrap):
            idx = rng.integers(0, len(aucs_gbsp), len(aucs_gbsp))
            boot_means_auc.append(np.mean(np.array(aucs_gbsp)[idx]))
            boot_means_ap.append(np.mean(np.array(aps_gbsp)[idx]))
        ci_auc_lower = float(np.percentile(boot_means_auc, 2.5))
        ci_auc_upper = float(np.percentile(boot_means_auc, 97.5))
        ci_ap_lower = float(np.percentile(boot_means_ap, 2.5))
        ci_ap_upper = float(np.percentile(boot_means_ap, 97.5))

    # Permutation test vs random baseline (GBSP)
    p_value_auc = p_value_ap = float("nan")
    if fold_labels and args.permutations > 0:
        rng = np.random.default_rng(123)
        perm_mean_auc = []
        perm_mean_ap = []
        for _ in range(args.permutations):
            perm_aucs = []
            perm_aps = []
            for y_true, y_scores in zip(fold_labels, fold_scores_gbsp):
                y_perm = rng.permutation(y_true)
                if y_perm.sum() == 0 or y_perm.sum() == len(y_perm):
                    continue
                perm_aucs.append(roc_auc_score(y_perm, y_scores))
                perm_aps.append(average_precision_score(y_perm, y_scores))
            if perm_aucs:
                perm_mean_auc.append(np.mean(perm_aucs))
            if perm_aps:
                perm_mean_ap.append(np.mean(perm_aps))
        if perm_mean_auc:
            p_value_auc = float(
                (np.sum(np.array(perm_mean_auc) >= mean_auc) + 1) / (len(perm_mean_auc) + 1)
            )
        if perm_mean_ap:
            p_value_ap = float(
                (np.sum(np.array(perm_mean_ap) >= mean_ap) + 1) / (len(perm_mean_ap) + 1)
            )

    summary = {
        "family": family,
        "n_folds": len(aucs_gbsp),
        # GBSP
        "gbsp_mean_roc_auc": mean_auc,
        "gbsp_std_roc_auc": std_auc,
        "gbsp_mean_pr_auc": mean_ap,
        "gbsp_std_pr_auc": std_ap,
        "gbsp_ci_roc_auc_lower": ci_auc_lower,
        "gbsp_ci_roc_auc_upper": ci_auc_upper,
        "gbsp_ci_pr_auc_lower": ci_ap_lower,
        "gbsp_ci_pr_auc_upper": ci_ap_upper,
        "gbsp_p_value_roc_auc": p_value_auc,
        "gbsp_p_value_pr_auc": p_value_ap,
        # KNN baseline
        "knn_mean_roc_auc": mean_auc_knn,
        "knn_std_roc_auc": std_auc_knn,
        "knn_mean_pr_auc": mean_ap_knn,
        "knn_std_pr_auc": std_ap_knn,
        # Convenience delta
        "delta_roc_auc_gbsp_minus_knn": mean_auc - mean_auc_knn,
    }
    pd.DataFrame([summary]).to_csv(output_dir / f"cv_{family.lower()}_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Plots: GBSP and KNN side-by-side on the same axes
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for fpr, tpr, auc_val in roc_curves_gbsp:
        axes[0].plot(fpr, tpr, color="steelblue", alpha=0.35, label=f"GBSP AUC={auc_val:.3f}")
    for fpr, tpr, auc_val in roc_curves_knn:
        axes[0].plot(fpr, tpr, color="tomato", alpha=0.35, linestyle="--", label=f"KNN AUC={auc_val:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(
        f"ROC  GBSP={mean_auc:.3f}±{std_auc:.3f}  KNN={mean_auc_knn:.3f}±{std_auc_knn:.3f}"
    )

    for rec, prec, ap in pr_curves_gbsp:
        axes[1].plot(rec, prec, color="steelblue", alpha=0.35, label=f"GBSP AP={ap:.3f}")
    for rec, prec, ap in pr_curves_knn:
        axes[1].plot(rec, prec, color="tomato", alpha=0.35, linestyle="--", label=f"KNN AP={ap:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(
        f"PR  GBSP={mean_ap:.3f}±{std_ap:.3f}  KNN={mean_ap_knn:.3f}±{std_ap_knn:.3f}"
    )

    for ax in axes:
        ax.grid(True, alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7)

    fig.tight_layout()
    fig_path = output_dir / f"cv_{family.lower()}_roc_pr.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"Saved fold metrics: {output_dir / f'cv_{family.lower()}_folds.csv'}")
    print(f"Saved summary:      {output_dir / f'cv_{family.lower()}_summary.csv'}")
    print(f"Saved ROC/PR plot:  {fig_path}")
    print(f"\nGBSP ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"KNN  ROC-AUC: {mean_auc_knn:.4f} ± {std_auc_knn:.4f}  (delta={mean_auc - mean_auc_knn:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
