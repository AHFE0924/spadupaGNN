#!/usr/bin/env python3
"""Cluster sequences at a specified identity threshold.

Clustering method priority (when --method auto):
  1. DIAMOND  -- BLOSUM-based; preferred because it accounts for conservative
                 substitutions at easily-mutated residues (see cluster_utils).
  2. cd-hit   -- fast identity clustering.
  3. greedy   -- pure-Python pairwise fallback; no external dependencies.
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

from cluster_utils import read_fasta_records, run_clustering, write_cluster_csv


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
        choices=["auto", "diamond", "cdhit", "greedy"],
        default="auto",
        help=(
            "Clustering method (default: auto). "
            "'auto' tries DIAMOND → cd-hit → greedy. "
            "'diamond' uses BLOSUM-based clustering (recommended). "
            "'cdhit' uses CD-HIT. "
            "'greedy' uses pure-Python pairwise alignment."
        ),
    )
    parser.add_argument(
        "--diamond-coverage",
        type=float,
        default=0.8,
        help="Minimum query/subject coverage for DIAMOND (default: 0.8)",
    )
    parser.add_argument(
        "--diamond-sensitivity",
        choices=["--sensitive", "--more-sensitive", "--very-sensitive"],
        default="--sensitive",
        help="DIAMOND sensitivity mode (default: --sensitive)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_prefix = Path(args.output)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    records = read_fasta_records(args.input)
    if not records:
        raise SystemExit("No sequences found in FASTA.")

    result = run_clustering(
        fasta_path=args.input,
        output_prefix=str(output_prefix),
        identity=args.identity,
        method=args.method,
    )

    cluster_csv = output_prefix.with_suffix(".csv")
    write_cluster_csv(result.assignments, str(cluster_csv))
    print(f"Cluster assignments saved to {cluster_csv}")
    if result.clstr_path:
        print(f"Cluster file: {result.clstr_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
