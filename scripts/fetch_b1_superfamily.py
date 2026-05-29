#!/usr/bin/env python3
"""Fetch B1 MBL superfamily sequences from UniProt and cluster at 40% identity."""
from __future__ import annotations

import argparse
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from Bio import SeqIO

from scripts.cluster_utils import greedy_cluster, run_cdhit, write_cluster_csv


DEFAULT_FAMILIES = ["NDM", "VIM", "IMP", "SPM", "GIM", "SIM"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch B1 MBL sequences from UniProt")
    parser.add_argument("--output", default="data/b1_superfamily.fasta", help="Output FASTA")
    parser.add_argument(
        "--families",
        default=",".join(DEFAULT_FAMILIES),
        help="Comma-separated family genes (default: NDM,VIM,IMP,SPM,GIM,SIM)",
    )
    parser.add_argument(
        "--reviewed",
        action="store_true",
        help="Restrict to reviewed (Swiss-Prot) entries",
    )
    parser.add_argument(
        "--taxon",
        default=None,
        help="Optional NCBI taxonomy ID filter (e.g., 2 for bacteria)",
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=None,
        help="Optional maximum number of sequences to keep",
    )
    parser.add_argument(
        "--cluster-identity",
        type=float,
        default=0.4,
        help="CD-HIT identity threshold (default: 0.4)",
    )
    parser.add_argument(
        "--cluster-output",
        default="output/clusters/b1_superfamily_40",
        help="Output prefix for clustering results",
    )
    parser.add_argument(
        "--split-dir",
        default=None,
        help="Optional directory to write per-family FASTA files",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional custom UniProt query string (overrides defaults)",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "cdhit", "greedy"],
        default="auto",
        help="Clustering method (default: auto)",
    )
    return parser.parse_args()


def build_query(families: List[str], reviewed: bool, taxon: Optional[str]) -> str:
    gene_query = " OR ".join([f"gene:{fam}" for fam in families])
    query = f"(({gene_query}) OR (protein_name:\"metallo-beta-lactamase\"))"
    if reviewed:
        query = f"{query} AND reviewed:true"
    if taxon:
        query = f"{query} AND taxonomy_id:{taxon}"
    return query


def fetch_fasta(query: str, output_path: Path) -> None:
    base = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "format": "fasta",
        "query": query,
    }
    url = f"{base}?{urllib.parse.urlencode(params)}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, output_path.open("wb") as handle:
        handle.write(response.read())


def family_from_header(header: str, families: List[str]) -> Optional[str]:
    text = header.upper()
    for fam in families:
        if re.search(rf"\b{re.escape(fam)}\b", text):
            return fam
    return None


def main() -> int:
    args = parse_args()
    families = [f.strip().upper() for f in args.families.split(",") if f.strip()]

    query = args.query or build_query(families, args.reviewed, args.taxon)
    output_path = Path(args.output)
    fetch_fasta(query, output_path)

    records = list(SeqIO.parse(str(output_path), "fasta"))
    if args.max_seqs:
        records = records[: args.max_seqs]
        SeqIO.write(records, str(output_path), "fasta")

    if args.split_dir:
        split_dir = Path(args.split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        by_family: Dict[str, List[SeqIO.SeqRecord]] = {fam: [] for fam in families}
        for rec in records:
            fam = family_from_header(rec.description, families)
            if fam:
                by_family[fam].append(rec)
        for fam, fam_records in by_family.items():
            if fam_records:
                SeqIO.write(fam_records, str(split_dir / f"{fam.lower()}.fasta"), "fasta")

    cluster_prefix = Path(args.cluster_output)
    cluster_prefix.parent.mkdir(parents=True, exist_ok=True)

    result = None
    if args.method in {"auto", "cdhit"}:
        try:
            result = run_cdhit(str(output_path), str(cluster_prefix), args.cluster_identity)
            print(f"cd-hit clustering complete: {result.cluster_count} clusters")
        except FileNotFoundError:
            if args.method == "cdhit":
                raise SystemExit("cd-hit not found. Install it or use --method greedy.")

    if result is None:
        result = greedy_cluster(records, args.cluster_identity)
        print(f"Greedy clustering complete: {result.cluster_count} clusters")

    cluster_csv = cluster_prefix.with_suffix(".csv")
    write_cluster_csv(result.assignments, str(cluster_csv))
    print(f"Saved clustered FASTA: {output_path}")
    print(f"Cluster assignments: {cluster_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
