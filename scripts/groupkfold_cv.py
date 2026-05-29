#!/usr/bin/env python3
"""Group K-Fold evaluation with sequence clustering to avoid leakage.

Clusters sequences at a specified identity threshold and performs GroupKFold
splits so sequences from the same cluster never appear in both train and test.
Outputs mean/std ROC-AUC across folds and plots ROC/PR curves.
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
    run_cdhit,
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
    propagated = (propagated - propagated.min()) / (propagated.max() - propagated.min() + 1e-8)
    return propagated


def compute_scores_from_train(
    reference: SeqIO.SeqRecord,
    train_records: Sequence[SeqIO.SeqRecord],
    embeddings: Dict[str, np.ndarray],
    alpha: float,
    hops: int,
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

    active_positions = np.array([max(0, n_residues // 3), max(0, n_residues // 2)], dtype=int)
    dist = np.array([min(abs(i - a) for a in active_positions) for i in range(n_residues)], dtype=np.float32)
    biophysical = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
    combined = 0.80 * propagated + 0.20 * biophysical

    return {
        "variance": var_norm,
        "graph": propagated,
        "combined": combined,
    }


def embed_sequences(
    records: Sequence[SeqIO.SeqRecord],
    device: str,
    batch_size: int,
    cache_path: Optional[str],
) -> Dict[str, np.ndarray]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GroupKFold CV with clustering")
    parser.add_argument("--input", required=True, help="Input FASTA file")
    parser.add_argument("--family", default="VIM", help="Family filter: NDM, VIM, IMP")
    parser.add_argument("--identity", type=float, default=0.3, help="Cluster identity threshold")
    parser.add_argument("--clusters", default=None, help="Optional cluster CSV file")
    parser.add_argument("--cluster-method", choices=["auto", "cdhit", "greedy"], default="auto")
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

    cluster_csv = None
    if args.clusters:
        cluster_csv = args.clusters
        assignments = load_cluster_csv(cluster_csv)
    else:
        cluster_csv = str(output_dir / f"clusters_{family.lower()}.csv")
        if args.cluster_method in {"auto", "cdhit"}:
            try:
                result = run_cdhit(args.input, str(output_dir / f"cdhit_{family.lower()}"), args.identity)
                assignments = result.assignments
                write_cluster_csv(assignments, cluster_csv)
            except FileNotFoundError:
                if args.cluster_method == "cdhit":
                    raise SystemExit("cd-hit not found. Install it or use --cluster-method greedy.")
                result = greedy_cluster(non_reference, args.identity)
                assignments = result.assignments
                write_cluster_csv(assignments, cluster_csv)
        else:
            result = greedy_cluster(non_reference, args.identity)
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

    embeddings = embed_sequences(records, device=device, batch_size=args.batch_size, cache_path=args.embed_cache)

    fold_rows = []
    roc_curves = []
    pr_curves = []
    aucs = []
    aps = []
    fold_labels = []
    fold_scores = []

    ref_seq = str(reference.seq)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(non_reference, groups=groups), start=1):
        train_records = [reference] + [non_reference[i] for i in train_idx]
        test_records = [non_reference[i] for i in test_idx]

        scores = compute_scores_from_train(reference, train_records, embeddings, alpha=args.alpha, hops=args.hops)

        positive_positions: List[int] = []
        for rec in test_records:
            positive_positions.extend(project_variant_positions(ref_seq, str(rec.seq)))
        positive_positions = sorted(set(positive_positions))

        y_true = np.array([1 if i in positive_positions else 0 for i in range(len(ref_seq))], dtype=int)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue

        roc_auc = roc_auc_score(y_true, scores["combined"])
        ap = average_precision_score(y_true, scores["combined"])
        fpr, tpr, _ = roc_curve(y_true, scores["combined"])
        prec, rec, _ = precision_recall_curve(y_true, scores["combined"])

        aucs.append(roc_auc)
        aps.append(ap)
        roc_curves.append((fpr, tpr, roc_auc))
        pr_curves.append((rec, prec, ap))
        fold_labels.append(y_true)
        fold_scores.append(scores["combined"])

        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": len(train_records),
                "n_test": len(test_records),
                "n_positive_positions": int(y_true.sum()),
                "roc_auc": roc_auc,
                "pr_auc": ap,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / f"cv_{family.lower()}_folds.csv", index=False)

    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    std_auc = float(np.std(aucs)) if aucs else float("nan")
    mean_ap = float(np.mean(aps)) if aps else float("nan")
    std_ap = float(np.std(aps)) if aps else float("nan")

    # Bootstrap CI for mean AUC/AP across folds
    ci_auc_lower = float("nan")
    ci_auc_upper = float("nan")
    ci_ap_lower = float("nan")
    ci_ap_upper = float("nan")
    if len(aucs) >= 2:
        rng = np.random.default_rng(42)
        boot_means_auc = []
        boot_means_ap = []
        for _ in range(args.bootstrap):
            idx = rng.integers(0, len(aucs), len(aucs))
            boot_means_auc.append(np.mean(np.array(aucs)[idx]))
            boot_means_ap.append(np.mean(np.array(aps)[idx]))
        ci_auc_lower = float(np.percentile(boot_means_auc, 2.5))
        ci_auc_upper = float(np.percentile(boot_means_auc, 97.5))
        ci_ap_lower = float(np.percentile(boot_means_ap, 2.5))
        ci_ap_upper = float(np.percentile(boot_means_ap, 97.5))

    # Permutation test vs random baseline (mean across folds)
    p_value_auc = float("nan")
    p_value_ap = float("nan")
    if fold_labels and args.permutations > 0:
        rng = np.random.default_rng(123)
        perm_mean_auc = []
        perm_mean_ap = []
        for _ in range(args.permutations):
            perm_aucs = []
            perm_aps = []
            for y_true, y_scores in zip(fold_labels, fold_scores):
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
            p_value_auc = float((np.sum(np.array(perm_mean_auc) >= mean_auc) + 1) / (len(perm_mean_auc) + 1))
        if perm_mean_ap:
            p_value_ap = float((np.sum(np.array(perm_mean_ap) >= mean_ap) + 1) / (len(perm_mean_ap) + 1))

    summary = {
        "family": family,
        "n_folds": len(aucs),
        "mean_roc_auc": mean_auc,
        "std_roc_auc": std_auc,
        "mean_pr_auc": mean_ap,
        "std_pr_auc": std_ap,
        "ci_roc_auc_lower": ci_auc_lower,
        "ci_roc_auc_upper": ci_auc_upper,
        "ci_pr_auc_lower": ci_ap_lower,
        "ci_pr_auc_upper": ci_ap_upper,
        "p_value_roc_auc": p_value_auc,
        "p_value_pr_auc": p_value_ap,
    }
    pd.DataFrame([summary]).to_csv(output_dir / f"cv_{family.lower()}_summary.csv", index=False)

    # Plot ROC and PR curves
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for fpr, tpr, auc_val in roc_curves:
        axes[0].plot(fpr, tpr, alpha=0.35, label=f"AUC={auc_val:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC (mean={mean_auc:.3f} ± {std_auc:.3f})")

    for rec, prec, ap in pr_curves:
        axes[1].plot(rec, prec, alpha=0.35, label=f"AP={ap:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR (mean={mean_ap:.3f} ± {std_ap:.3f})")

    for ax in axes:
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig_path = output_dir / f"cv_{family.lower()}_roc_pr.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"Saved fold metrics to {output_dir / f'cv_{family.lower()}_folds.csv'}")
    print(f"Saved summary to {output_dir / f'cv_{family.lower()}_summary.csv'}")
    print(f"Saved ROC/PR plot to {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
