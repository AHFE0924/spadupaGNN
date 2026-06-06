#!/usr/bin/env python3
"""Utility functions for sequence clustering.

Supports DIAMOND (recommended), cd-hit, or a greedy Biopython alignment
fallback for smaller datasets.

Clustering method priority (when method="auto"):
  1. DIAMOND  -- BLOSUM-based, handles variable/mutatable residues correctly
  2. cd-hit   -- fast identity clustering, good fallback if DIAMOND absent
  3. greedy   -- pure-Python pairwise alignment, slow but zero dependencies
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from Bio import SeqIO
from Bio.Align import PairwiseAligner


@dataclass
class ClusterResult:
    assignments: Dict[str, int]
    cluster_count: int
    method: str
    clstr_path: Optional[Path] = None


def read_fasta_records(fasta_path: str) -> List[SeqIO.SeqRecord]:
    return list(SeqIO.parse(fasta_path, "fasta"))


def sequence_identity(seq_a: str, seq_b: str) -> float:
    """Compute global alignment identity as matches / alignment length."""
    if seq_a == seq_b:
        return 1.0
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = 0.0
    aligner.extend_gap_score = 0.0
    alignment = aligner.align(seq_a, seq_b)[0]
    matches = sum(a == b for a, b in zip(*alignment))
    return matches / max(1, alignment.length)


def greedy_cluster(records: Iterable[SeqIO.SeqRecord], identity: float) -> ClusterResult:
    clusters: List[Tuple[str, str]] = []
    assignments: Dict[str, int] = {}

    for rec in records:
        seq = str(rec.seq)
        assigned = False
        for cluster_id, (rep_id, rep_seq) in enumerate(clusters):
            if sequence_identity(seq, rep_seq) >= identity:
                assignments[rec.id] = cluster_id
                assigned = True
                break
        if not assigned:
            clusters.append((rec.id, seq))
            assignments[rec.id] = len(clusters) - 1

    return ClusterResult(assignments=assignments, cluster_count=len(clusters), method="greedy")


# ---------------------------------------------------------------------------
# DIAMOND clustering
# ---------------------------------------------------------------------------

def run_diamond(
    fasta_path: str,
    output_prefix: str,
    identity: float,
    coverage: float = 0.8,
    sensitivity: str = "--sensitive",
) -> ClusterResult:
    """Cluster sequences using DIAMOND's BLOSUM-based all-vs-all search.

    DIAMOND is preferred over CD-HIT because its BLOSUM62 scoring matrix
    accounts for biochemically conservative substitutions at easily-mutated
    residues -- directly relevant for metallo-beta-lactamase variant analysis.
    See: Buchfink et al. (2021) Nature Methods; precedent in MBL literature.

    Parameters
    ----------
    fasta_path:  Input FASTA file.
    output_prefix: Path prefix for intermediate and output files.
    identity:    Minimum sequence identity (0–1) for two sequences to share
                 a cluster.  Translated to a percentage for DIAMOND's
                 --id flag.
    coverage:    Minimum query/subject coverage (0–1).  Default 0.8.
    sensitivity: DIAMOND sensitivity flag.  '--sensitive' is a good default;
                 use '--more-sensitive' for highly diverged sequences.

    Returns
    -------
    ClusterResult with method="diamond".
    """
    diamond = shutil.which("diamond")
    if not diamond:
        raise FileNotFoundError("DIAMOND not found on PATH. Install with: conda install -c bioconda diamond")

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    db_path = prefix.with_suffix(".dmnd")
    hits_path = prefix.with_suffix(".tsv")

    # Build DIAMOND protein database
    subprocess.check_call(
        [diamond, "makedb", "--in", fasta_path, "--db", str(db_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # All-vs-all search with BLOSUM62
    subprocess.check_call(
        [
            diamond, "blastp",
            "--db", str(db_path),
            "--query", fasta_path,
            "--out", str(hits_path),
            "--outfmt", "6", "qseqid", "sseqid", "pident", "qcovhsp", "scovhsp",
            "--id", str(identity * 100),
            "--query-cover", str(coverage * 100),
            "--subject-cover", str(coverage * 100),
            sensitivity,
            "--no-self-hits",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    assignments = _parse_diamond_hits_to_clusters(hits_path, fasta_path)
    return ClusterResult(
        assignments=assignments,
        cluster_count=len(set(assignments.values())),
        method="diamond",
        clstr_path=hits_path,
    )


def _parse_diamond_hits_to_clusters(hits_path: Path, fasta_path: str) -> Dict[str, int]:
    """Single-linkage clustering from DIAMOND all-vs-all hits."""
    # Build adjacency list from hits
    edges: Dict[str, List[str]] = {}
    with hits_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            q, s = parts[0], parts[1]
            edges.setdefault(q, []).append(s)
            edges.setdefault(s, []).append(q)

    # Union-Find for connected components
    records = read_fasta_records(fasta_path)
    all_ids = [rec.id for rec in records]
    parent = {seq_id: seq_id for seq_id in all_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    for node, neighbors in edges.items():
        if node not in parent:
            continue
        for nbr in neighbors:
            if nbr in parent:
                union(node, nbr)

    # Assign integer cluster IDs
    root_to_id: Dict[str, int] = {}
    assignments: Dict[str, int] = {}
    for seq_id in all_ids:
        root = find(seq_id)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
        assignments[seq_id] = root_to_id[root]

    return assignments


# ---------------------------------------------------------------------------
# CD-HIT clustering (kept as secondary fallback)
# ---------------------------------------------------------------------------

def _cdhit_word_length(identity: float) -> int:
    if identity >= 0.7:
        return 5
    if identity >= 0.6:
        return 4
    if identity >= 0.5:
        return 3
    return 2


def run_cdhit(fasta_path: str, output_prefix: str, identity: float) -> ClusterResult:
    cdhit = shutil.which("cd-hit")
    if not cdhit:
        raise FileNotFoundError("cd-hit not found on PATH")

    out_path = Path(output_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    word_len = _cdhit_word_length(identity)

    cmd = [
        cdhit,
        "-i", str(fasta_path),
        "-o", str(out_path),
        "-c", str(identity),
        "-n", str(word_len),
        "-d", "0",
    ]
    subprocess.check_call(cmd)
    clstr_path = out_path.with_suffix(out_path.suffix + ".clstr")
    assignments = parse_cdhit_clstr(clstr_path)
    return ClusterResult(
        assignments=assignments,
        cluster_count=len(set(assignments.values())),
        method="cd-hit",
        clstr_path=clstr_path,
    )


def parse_cdhit_clstr(clstr_path: Path) -> Dict[str, int]:
    assignments: Dict[str, int] = {}
    cluster_id: Optional[int] = None
    with clstr_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">Cluster"):
                cluster_id = int(line.split()[1])
                continue
            if cluster_id is None:
                continue
            match = re.search(r">([^\.\s]+)", line)
            if match:
                seq_id = match.group(1)
                assignments[seq_id] = cluster_id
    return assignments


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def run_clustering(
    fasta_path: str,
    output_prefix: str,
    identity: float,
    method: str = "auto",
) -> ClusterResult:
    """Dispatch to DIAMOND → CD-HIT → greedy depending on availability.

    Parameters
    ----------
    method: 'auto' tries DIAMOND first, then cd-hit, then greedy.
            'diamond', 'cdhit', or 'greedy' forces a specific tool.
    """
    if method in {"auto", "diamond"}:
        try:
            result = run_diamond(fasta_path, output_prefix, identity)
            print(f"DIAMOND clustering complete: {result.cluster_count} clusters")
            return result
        except FileNotFoundError:
            if method == "diamond":
                raise SystemExit(
                    "DIAMOND not found. Install with: conda install -c bioconda diamond"
                )

    if method in {"auto", "cdhit"}:
        try:
            result = run_cdhit(fasta_path, output_prefix, identity)
            print(f"CD-HIT clustering complete: {result.cluster_count} clusters")
            return result
        except FileNotFoundError:
            if method == "cdhit":
                raise SystemExit("cd-hit not found. Install it or use --method greedy.")

    records = read_fasta_records(fasta_path)
    result = greedy_cluster(records, identity)
    print(f"Greedy clustering complete: {result.cluster_count} clusters")
    return result


def write_cluster_csv(assignments: Dict[str, int], output_csv: str) -> None:
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["sequence_id,cluster_id"]
    for seq_id, cluster_id in sorted(assignments.items(), key=lambda x: (x[1], x[0])):
        lines.append(f"{seq_id},{cluster_id}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def load_cluster_csv(path: str) -> Dict[str, int]:
    assignments: Dict[str, int] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            seq_id, cluster_id = line.strip().split(",")
            assignments[seq_id] = int(cluster_id)
    return assignments
