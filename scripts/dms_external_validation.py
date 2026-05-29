#!/usr/bin/env python3
"""Validate predictions against an external DMS dataset without retraining."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="External DMS validation")
    parser.add_argument("--dms", required=True, help="Path to DMS dataset (CSV/TSV)")
    parser.add_argument("--predictions", required=True, help="Prediction CSV (mutation scores)")
    parser.add_argument("--dms-score", default="score", help="Column with DMS score")
    parser.add_argument("--dms-label", default=None, help="Optional binary label column")
    parser.add_argument("--mutation-column", default=None, help="Mutation string column (e.g., A123V)")
    parser.add_argument("--position-column", default=None, help="Position column (1-indexed by default)")
    parser.add_argument("--position-0-indexed", action="store_true", help="Treat position column as 0-indexed")
    parser.add_argument("--pred-position", default="Residue_Index", help="Position column in prediction CSV")
    parser.add_argument("--pred-score", default="Combined_Score", help="Score column in prediction CSV")
    parser.add_argument("--invert-dms", action="store_true", help="Invert DMS score if higher is worse")
    parser.add_argument("--threshold", type=float, default=None, help="Optional threshold to binarize DMS scores")
    return parser.parse_args()


def parse_position_from_mutation(mutation: str) -> Optional[int]:
    match = re.search(r"(\d+)", str(mutation))
    if not match:
        return None
    return int(match.group(1))


def main() -> int:
    args = parse_args()

    dms_path = Path(args.dms)
    dms_df = pd.read_csv(dms_path, sep=None, engine="python")

    if args.position_column:
        pos = dms_df[args.position_column].astype(int)
        if not args.position_0_indexed:
            pos = pos - 1
    elif args.mutation_column:
        pos = dms_df[args.mutation_column].apply(parse_position_from_mutation)
        if pos.isnull().any():
            raise SystemExit("Could not parse positions from mutation column.")
        pos = pos.astype(int) - 1
    else:
        raise SystemExit("Provide either --position-column or --mutation-column.")

    dms_df["_pos0"] = pos

    pred_df = pd.read_csv(args.predictions)
    pred_df["_pos0"] = pred_df[args.pred_position].astype(int)

    merged = dms_df.merge(pred_df[["_pos0", args.pred_score]], on="_pos0", how="inner")
    if merged.empty:
        raise SystemExit("No overlapping positions between DMS and predictions.")

    y_scores = merged[args.pred_score].values
    dms_scores = merged[args.dms_score].values
    if args.invert_dms:
        dms_scores = -dms_scores

    spearman = spearmanr(dms_scores, y_scores, nan_policy="omit")
    pearson = pearsonr(dms_scores, y_scores)

    results = {
        "n_points": int(len(merged)),
        "spearman_r": float(spearman.correlation),
        "spearman_p": float(spearman.pvalue),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
    }

    if args.dms_label and args.dms_label in merged.columns:
        y_true = merged[args.dms_label].astype(int).values
        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            results["roc_auc"] = float(roc_auc_score(y_true, y_scores))
            results["pr_auc"] = float(average_precision_score(y_true, y_scores))
    elif args.threshold is not None:
        y_true = (dms_scores >= args.threshold).astype(int)
        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            results["roc_auc"] = float(roc_auc_score(y_true, y_scores))
            results["pr_auc"] = float(average_precision_score(y_true, y_scores))

    out_path = Path(args.dms).with_suffix(".validation.csv")
    pd.DataFrame([results]).to_csv(out_path, index=False)
    print(f"Saved validation summary: {out_path}")
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
