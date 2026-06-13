#!/usr/bin/env python3
"""Fetch B1 MBL superfamily sequences from UniProt and cluster.

Clustering method priority (when --method auto):
  1. DIAMOND  -- BLOSUM-based; recommended because it accounts for
                 conservative substitutions at easily-mutated residues,
                 which is especially important for metallo-beta-lactamase
                 variant analysis.  See cluster_utils.run_diamond for refs.
  2. cd-hit   -- fast identity clustering (legacy default).
  3. greedy   -- pure-Python pairwise fallback; no external dependencies.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Bio import SeqIO

# Ensure repo root and scripts directory are on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cluster_utils import greedy_cluster, run_clustering, write_cluster_csv


DEFAULT_FAMILIES = ["NDM", "VIM", "IMP", "SPM", "GIM", "SIM"]
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch B1 MBL sequences from UniProt")
    parser.add_argument("--output", default="data/b1_superfamily.fasta", help="Output FASTA")
    parser.add_argument(
        "--raw-output",
        default=None,
        help="Optional raw FASTA output (keeps unfiltered download)",
    )
    parser.add_argument(
        "--families",
        default=",".join(DEFAULT_FAMILIES),
        help="Comma-separated family genes (default: NDM,VIM,IMP,SPM,GIM,SIM)",
    )
    parser.add_argument(
        "--families-file",
        default=None,
        help="Optional file with one family gene per line",
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
        "--min-length",
        type=int,
        default=None,
        help="Optional minimum sequence length",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional maximum sequence length",
    )
    parser.add_argument(
        "--exclude-fragments",
        action="store_true",
        help="Exclude sequences labeled as fragment/partial",
    )
    parser.add_argument(
        "--max-ambiguous",
        type=int,
        default=None,
        help="Maximum number of non-standard residues (e.g., X).",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Deduplicate exact sequences",
    )
    parser.add_argument(
        "--cluster-identity",
        type=float,
        default=0.4,
        help="Clustering identity threshold (default: 0.4)",
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
        "--extra-query",
        default=None,
        help="Optional query to AND with the default query",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "diamond", "cdhit", "greedy"],
        default="auto",
        help=(
            "Clustering method (default: auto). "
            "'auto' tries DIAMOND → cd-hit → greedy. "
            "'diamond' is recommended (BLOSUM-based, handles easily-mutated residues). "
            "'cdhit' uses CD-HIT sequence identity. "
            "'greedy' uses pure-Python pairwise alignment."
        ),
    )
    parser.add_argument(
        "--stats-output",
        default="output/b1_superfamily_stats.json",
        help="Write dataset/cluster stats JSON (default: output/b1_superfamily_stats.json)",
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


def fetch_fasta(query: str, output_path: Path, retries: int = 5, backoff: float = 10.0) -> None:
    import time
    base = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "format": "fasta",
        "query": query,
    }
    url = f"{base}?{urllib.parse.urlencode(params)}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, output_path.open("wb") as handle:
                handle.write(response.read())
            return
        except Exception as exc:
            if attempt == retries:
                raise
            wait = backoff * attempt
            print(f"Fetch attempt {attempt} failed ({exc}). Retrying in {wait:.0f}s...")
            time.sleep(wait)


def family_from_header(header: str, families: List[str]) -> Optional[str]:
    text = header.upper()
    for fam in families:
        if re.search(rf"\b{re.escape(fam)}\b", text):
            return fam
    return None


def is_fragment(description: str) -> bool:
    return bool(re.search(r"fragment|partial|truncated", description, re.IGNORECASE))


def count_ambiguous(seq: str) -> int:
    return sum(1 for aa in seq if aa not in STANDARD_AA)


def filter_records(
    records: List[SeqIO.SeqRecord],
    min_length: Optional[int],
    max_length: Optional[int],
    max_ambiguous: Optional[int],
    exclude_fragments: bool,
    dedup: bool,
) -> Tuple[List[SeqIO.SeqRecord], Dict[str, int]]:
    removed = {
        "too_short": 0,
        "too_long": 0,
        "fragment": 0,
        "ambiguous": 0,
        "duplicate": 0,
    }
    seen: Dict[str, str] = {}
    filtered: List[SeqIO.SeqRecord] = []

    for rec in records:
        seq = str(rec.seq).upper()
        if min_length is not None and len(seq) < min_length:
            removed["too_short"] += 1
            continue
        if max_length is not None and len(seq) > max_length:
            removed["too_long"] += 1
            continue
        if exclude_fragments and is_fragment(rec.description):
            removed["fragment"] += 1
            continue
        if max_ambiguous is not None and count_ambiguous(seq) > max_ambiguous:
            removed["ambiguous"] += 1
            continue
        if dedup:
            if seq in seen:
                removed["duplicate"] += 1
                continue
            seen[seq] = rec.id
        filtered.append(rec)

    return filtered, removed


def summarize_records(
    records: List[SeqIO.SeqRecord], families: List[str]
) -> Dict[str, object]:
    lengths = [len(str(rec.seq)) for rec in records]
    length_stats: Dict[str, float] = {}
    if lengths:
        length_stats = {
            "min": float(min(lengths)),
            "max": float(max(lengths)),
            "mean": float(statistics.mean(lengths)),
            "median": float(statistics.median(lengths)),
        }

    family_counts: Dict[str, int] = {fam: 0 for fam in families}
    other_count = 0
    for rec in records:
        fam = family_from_header(rec.description, families)
        if fam:
            family_counts[fam] += 1
        else:
            other_count += 1

    return {
        "length": length_stats,
        "family_counts": family_counts,
        "other_count": other_count,
    }


def summarize_clusters(assignments: Dict[str, int]) -> Dict[str, object]:
    sizes: Dict[int, int] = {}
    for cluster_id in assignments.values():
        sizes[cluster_id] = sizes.get(cluster_id, 0) + 1
    size_list = list(sizes.values())
    size_hist: Dict[str, int] = {}
    for size in size_list:
        size_hist[str(size)] = size_hist.get(str(size), 0) + 1

    stats: Dict[str, object] = {"cluster_count": len(sizes), "size_histogram": size_hist}
    if size_list:
        stats.update(
            {
                "min_size": int(min(size_list)),
                "max_size": int(max(size_list)),
                "mean_size": float(statistics.mean(size_list)),
                "median_size": float(statistics.median(size_list)),
            }
        )
    return stats


def write_family_counts_csv(family_counts: Dict[str, int], other_count: int, path: Path) -> None:
    lines = ["family,count"]
    for fam, count in sorted(family_counts.items()):
        lines.append(f"{fam},{count}")
    lines.append(f"OTHER,{other_count}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    families = [f.strip().upper() for f in args.families.split(",") if f.strip()]
    if args.families_file:
        file_families = Path(args.families_file).read_text(encoding="utf-8").splitlines()
        families.extend([f.strip().upper() for f in file_families if f.strip()])
        families = sorted(set(families))

    query = args.query or build_query(families, args.reviewed, args.taxon)
    if args.extra_query:
        query = f"({query}) AND ({args.extra_query})"

    output_path = Path(args.output)
    raw_output = Path(args.raw_output) if args.raw_output else output_path
    fetch_fasta(query, raw_output)

    records = list(SeqIO.parse(str(raw_output), "fasta"))
    filtered_records, removed = filter_records(
        records,
        min_length=args.min_length,
        max_length=args.max_length,
        max_ambiguous=args.max_ambiguous,
        exclude_fragments=args.exclude_fragments,
        dedup=args.dedup,
    )
    if args.max_seqs:
        filtered_records = filtered_records[: args.max_seqs]

    if raw_output != output_path or filtered_records != records or args.max_seqs:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        SeqIO.write(filtered_records, str(output_path), "fasta")

    if args.split_dir:
        split_dir = Path(args.split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        by_family: Dict[str, List[SeqIO.SeqRecord]] = {fam: [] for fam in families}
        for rec in filtered_records:
            fam = family_from_header(rec.description, families)
            if fam:
                by_family[fam].append(rec)
        for fam, fam_records in by_family.items():
            if fam_records:
                SeqIO.write(fam_records, str(split_dir / f"{fam.lower()}.fasta"), "fasta")

    cluster_prefix = Path(args.cluster_output)
    cluster_prefix.parent.mkdir(parents=True, exist_ok=True)

    result = run_clustering(
        fasta_path=str(output_path),
        output_prefix=str(cluster_prefix),
        identity=args.cluster_identity,
        method=args.method,
    )

    cluster_csv = cluster_prefix.with_suffix(".csv")
    write_cluster_csv(result.assignments, str(cluster_csv))

    stats = {
        "raw_records": len(records),
        "filtered_records": len(filtered_records),
        "removed": removed,
        "clustering_method": result.method,
        "filters": {
            "min_length": args.min_length,
            "max_length": args.max_length,
            "max_ambiguous": args.max_ambiguous,
            "exclude_fragments": args.exclude_fragments,
            "dedup": args.dedup,
        },
    }
    stats.update(summarize_records(filtered_records, families))
    stats["clusters"] = summarize_clusters(result.assignments)

    stats_path = Path(args.stats_output) if args.stats_output else None
    if stats_path:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        family_csv = stats_path.with_suffix(".families.csv")
        write_family_counts_csv(stats["family_counts"], stats["other_count"], family_csv)
        print(f"Saved stats: {stats_path}")
        print(f"Saved family counts: {family_csv}")

    print(f"Saved clustered FASTA: {output_path}")
    print(f"Cluster assignments ({result.method}): {cluster_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
