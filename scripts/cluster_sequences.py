#!/usr/bin/env python3
"""Cluster sequences at a specified identity threshold.

Uses cd-hit when available; falls back to a greedy Biopython alignment strategy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cluster_utils import (
    greedy_cluster,
    read_fasta_records,
    run_cdhit,
    write_cluster_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster sequences at a target identity")
    parser.add_argument("--input", required=True, help="Input FASTA file")
    parser.add_argument(
        "--identity",
        type=float,
        default=0.3,
        help="Sequence identity threshold (default: 0.3)",
    )
    parser.add_argument(
        "--output",
        default="output/clusters",
        help="Output prefix or folder for clusters (default: output/clusters)",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "cdhit", "greedy"],
        default="auto",
        help="Clustering method (default: auto)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_prefix = Path(args.output)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    records = read_fasta_records(args.input)
    if not records:
        raise SystemExit("No sequences found in FASTA.")

    result = None
    if args.method in {"auto", "cdhit"}:
        try:
            result = run_cdhit(args.input, str(output_prefix), args.identity)
            print(f"cd-hit clustering complete: {result.cluster_count} clusters")
        except FileNotFoundError:
            if args.method == "cdhit":
                raise SystemExit("cd-hit not found. Install it or use --method greedy.")

    if result is None:
        result = greedy_cluster(records, args.identity)
        print(f"Greedy clustering complete: {result.cluster_count} clusters")

    cluster_csv = output_prefix.with_suffix(".csv")
    write_cluster_csv(result.assignments, str(cluster_csv))
    print(f"Cluster assignments saved to {cluster_csv}")
    if result.clstr_path:
        print(f"cd-hit cluster file: {result.clstr_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
