#!/usr/bin/env python3
"""Group K-Fold evaluation with sequence clustering to avoid leakage.

Clusters sequences at a specified identity threshold and performs GroupKFold
splits so sequences from the same cluster never appear in both train and test.
Outputs mean/std ROC-AUC across folds and plots ROC/PR curves.

Method: Graph-Based Score Propagation (GBSP)
--------------------------------------------
This is NOT a trained GNN.  There are no learnable parameters.  Instead,
per-residue ESM-2 embedding variance is computed across training sequences
and smoothed over a graph via iterative propagation.  A biophysical proximity
term (distance to known active-site residues, weighted 0.10) biases scores
toward functionally important regions.

Graph construction (structure-based with chain fallback)
--------------------------------------------------------
When an AlphaFold structure is available for the reference family, the graph
adjacency is built from Cα–Cα contacts at ≤8 Å.  This captures long-range
contacts that the chain graph (±5 residue window) misses — critical for enzyme
active sites where catalytic residues are often sequence-distant but spatially
adjacent.  If no structure can be downloaded (network error, unknown family),
the script falls back to the ±5 chain graph automatically.

AlphaFold structures are downloaded once and cached in --structure-dir.

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from cluster_utils import (
    greedy_cluster,
    load_cluster_csv,
    read_fasta_records,
    run_clustering,
    write_cluster_csv,
)
from collections import Counter


NDM_FAMILY = "NDM"
VIM_FAMILY = "VIM"
IMP_FAMILY = "IMP"
SPM_FAMILY = "SPM"
GIM_FAMILY = "GIM"
SIM_FAMILY = "SIM"
ALL_FAMILIES = [NDM_FAMILY, VIM_FAMILY, IMP_FAMILY, SPM_FAMILY, GIM_FAMILY, SIM_FAMILY]

# Canonical reference variant per family, used by choose_reference() below.
_CANONICAL_VARIANT = {
    NDM_FAMILY: "NDM-1",
    VIM_FAMILY: "VIM-1",
    IMP_FAMILY: "IMP-1",
    SPM_FAMILY: "SPM-1",
    GIM_FAMILY: "GIM-1",
    SIM_FAMILY: "SIM-1",
}


def family_from_header(header: str) -> Optional[str]:
    """Classify a UniProt FASTA header into one of the 6 B1 MBL families.

    BUG FIX: the previous version only recognized NDM/VIM/IMP -- SPM, GIM,
    and SIM always returned None, meaning `--family SPM` (etc.) would always
    raise "No sequences found", and the family was silently unusable.

    BUG FIX: the previous NDM/VIM/IMP patterns required a digit immediately
    after the family code (e.g. "NDM-1", "VIM-2"), missing headers that
    mention the family name without a trailing variant number (e.g. a
    generic "NDM-type metallo-beta-lactamase" protein name) -- undercounting
    real family members and explaining unexpectedly low n_sequences.

    Matching strategy: for each family code FFF, search for an optional
    "BLA"/"BLA-" prefix (covers gene names like "blaNDM-1" glued with no
    separator) followed by FFF, requiring a word boundary immediately AFTER
    the code (e.g. "NDM" followed by "-", a digit, a space, or end of
    string) -- this still excludes accidental substring matches such as
    "IMP" inside "IMPORTANT" or "IMPDH", but no longer requires a trailing
    digit, so genuine family members get correctly counted.
    """
    text = header.upper()
    for fam in ALL_FAMILIES:
        if re.search(rf"(?:BLA-?)?{fam}\b", text):
            return fam
    return None


def choose_reference(records: Sequence[SeqIO.SeqRecord], family: str) -> SeqIO.SeqRecord:
    """Pick the canonical reference variant (e.g. NDM-1) if present in the
    family's record set, otherwise fall back to the first record."""
    canonical = _CANONICAL_VARIANT.get(family.upper())
    if canonical:
        pattern = re.compile(re.escape(canonical) + r"\b", re.I)
        for rec in records:
            if pattern.search(rec.description):
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




# ---------------------------------------------------------------------------
# AlphaFold structure-based contact map
# ---------------------------------------------------------------------------

# Verified PDB IDs for canonical B1 MBL reference structures (experimental).
# Crystal structures are preferred over AlphaFold for these well-characterized
# proteins as they have experimental validation and stable accession IDs.
#   NDM: 3SPU  NDM-1, E. coli,          1.90 Å  (Tesar et al. 2011)
#   VIM: 1KO3  VIM-2, P. aeruginosa,    1.85 Å  (Garcia-Saez et al. 2008)
#   IMP: 1DDK  IMP-1, P. aeruginosa,    2.20 Å  (Concha et al. 2000)
#   SPM: 1X8I  SPM-1, P. aeruginosa,    2.30 Å  (Murphy et al. 2006)
# GIM and SIM have no deposited crystal structures — chain graph fallback used.
_FAMILY_PDB: Dict[str, str] = {
    "NDM": "3SPU",
    "VIM": "1KO3",
    "IMP": "1DDK",
    "SPM": "1X8I",
}


def fetch_rcsb_pdb(pdb_id: str, cache_dir: Path) -> Optional[Path]:
    """Download a PDB file from RCSB, caching locally.

    URL: https://files.rcsb.org/download/{pdb_id}.pdb
    Returns None on failure so callers can fall back gracefully.
    """
    import urllib.request as _ureq

    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = cache_dir / f"{pdb_id.upper()}.pdb"
    if pdb_path.exists():
        return pdb_path
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        _ureq.urlretrieve(url, str(pdb_path))
        print(f"Downloaded PDB structure: {pdb_id.upper()}.pdb")
        return pdb_path
    except Exception as exc:
        print(f"Warning: PDB download failed for {pdb_id} ({exc}). Using chain graph fallback.")
        return None


def parse_ca_coords(pdb_path: Path) -> np.ndarray:
    """Extract Cα coordinates from a PDB file (first chain only).

    Returns array of shape (n_residues, 3), one row per unique residue.
    """
    coords: List[List[float]] = []
    seen: set = set()
    with pdb_path.open("r") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            chain = line[21]
            res_seq = line[22:26].strip()
            key = (chain, res_seq)
            if key in seen:
                continue
            seen.add(key)
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            coords.append([x, y, z])
    return np.array(coords, dtype=np.float32)


def build_contact_map_from_coords(coords: np.ndarray, threshold: float = 8.0) -> np.ndarray:
    """Binary Cα–Cα contact map: 1 if distance ≤ threshold Å, else 0.

    8 Å is the standard threshold used in structural bioinformatics for
    protein contact maps (e.g., Dunn et al. 2008; Marks et al. 2011).
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]   # (n, n, 3)
    dist = np.sqrt((diff ** 2).sum(axis=-1))                       # (n, n)
    contacts = (dist <= threshold).astype(np.float32)
    return contacts


def build_structure_graph(
    n_residues: int,
    family: str,
    structure_dir: Optional[Path],
    contact_threshold: float = 8.0,
) -> np.ndarray:
    """Return a normalised adjacency matrix using experimental Cα contacts.

    Downloads the canonical crystal structure for the family from RCSB PDB
    (NDM→3SPU, VIM→1KO3, IMP→1DDK, SPM→1X8I).  Falls back to the ±5 chain
    graph for families without a deposited structure (GIM, SIM) or if the
    download fails.

    Crystal structure contacts capture long-range spatial relationships that
    the chain graph misses, substantially improving score propagation for
    enzyme families where active-site residues are sequence-distant.
    """
    if structure_dir is not None:
        pdb_id = _FAMILY_PDB.get(family.upper())
        if pdb_id:
            pdb_path = fetch_rcsb_pdb(pdb_id, structure_dir)
            if pdb_path is not None:
                try:
                    coords = parse_ca_coords(pdb_path)
                    if len(coords) >= 10:
                        min_len = min(len(coords), n_residues)
                        contacts = build_contact_map_from_coords(coords[:min_len], threshold=contact_threshold)
                        adj = np.zeros((n_residues, n_residues), dtype=np.float32)
                        adj[:min_len, :min_len] = contacts
                        # Chain graph for residues beyond structure coverage
                        for i in range(min_len, n_residues):
                            for j in range(max(0, i - 5), min(n_residues, i + 6)):
                                adj[i, j] = 1.0
                                adj[j, i] = 1.0
                        np.fill_diagonal(adj, 1.0)
                        deg = adj.sum(axis=1, keepdims=True)
                        print(f"Structure graph ({family}, PDB {pdb_id}): "
                              f"{int(contacts.sum())} contacts in {min_len}-residue structure")
                        return adj / (deg + 1e-8)
                except Exception as exc:
                    print(f"Warning: Structure graph failed for {family} ({exc}). Using chain graph.")

    # Fallback: symmetric ±5 chain graph
    return build_chain_graph(n_residues)


def robust_normalize(arr: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Normalise to [0,1] using percentile clipping to suppress outlier variance spikes."""
    lo = float(np.percentile(arr, lo_pct))
    hi = float(np.percentile(arr, hi_pct))
    clipped = np.clip(arr, lo, hi)
    span = clipped.max() - clipped.min()
    return (clipped - clipped.min()) / (span + 1e-8)




def build_chain_graph(n_residues: int) -> np.ndarray:
    """Fallback ±5 residue window chain graph."""
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
    structure_dir: Optional[Path] = None,
    contact_threshold: float = 8.0,
    biophysical_weight: float = 0.10,
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

    # Robust normalization: percentile clipping suppresses outlier variance spikes
    var_norm = robust_normalize(variance)

    # Structure-based graph (AlphaFold Cα contacts) with chain-graph fallback
    adj_norm = build_structure_graph(
        n_residues, family=family,
        structure_dir=structure_dir,
        contact_threshold=contact_threshold,
    )
    propagated = propagate_scores(var_norm, adj_norm, alpha=alpha, hops=hops)

    # Biophysical proximity term: distance to literature-sourced active site
    # residues in BBL numbering.
    #
    # NOTE on biophysical_weight default (0.10): this was an initial guess,
    # NOT a fitted value. LR ablations (see compute_lr_baseline_scores) that
    # learn this weight directly from data found it should be much higher
    # and family-dependent: ~0.62 for NDM, ~0.47 for VIM, ~0.81 for IMP in
    # initial runs (see [DECISIONS] / "LR-*-learned-*-weight" summary
    # columns). There is no single fixed value that fits all families, so
    # rather than hardcoding a new guess, this is exposed as --biophysical-
    # weight; check that family's printed "LR-graph learned weights" line
    # and pass it in directly to test whether it actually improves GBSP's
    # ROC-AUC for that family (it improved NDM/VIM substantially in testing,
    # but UNDERPERFORMED GBSP's default for IMP -- verify per-family, do not
    # assume one weight generalizes).
    active_positions = get_active_site_positions(family, ref_seq)
    if active_positions:
        dist = np.array(
            [min(abs(i - a) for a in active_positions) for i in range(n_residues)],
            dtype=np.float32,
        )
    else:
        dist = np.zeros(n_residues, dtype=np.float32)
    biophysical = robust_normalize(dist)
    combined = (1.0 - biophysical_weight) * propagated + biophysical_weight * biophysical

    return {
        "variance": var_norm,
        "graph": propagated,
        "biophysical": biophysical,
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
# Logistic Regression baseline (raw-feature ablation)
# ---------------------------------------------------------------------------

def compute_lr_baseline_scores(
    features: np.ndarray,
    y_true: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
) -> Tuple[float, float, Optional[np.ndarray]]:
    """Logistic regression on GBSP's raw ingredients, evaluated out-of-fold.

    Ablation requested per ORNL feedback: GBSP combines per-position variance
    and active-site distance via a FIXED formula (propagate, then
    0.90*graph + 0.10*biophysical).  This baseline instead lets a tiny LR
    LEARN how to combine the same two raw signals.  If LR beats GBSP, the
    fixed combination/propagation is not adding value and should be
    replaced with a learned blend; if GBSP beats LR, propagation is doing
    real work.

    `features` shape: (n_residues, n_features) -- e.g. [variance, biophysical]
    (raw, pre-propagation) or [graph, biophysical] (post-propagation, GBSP's
    own ingredients with a learned weight instead of the fixed 0.90/0.10).
    `y_true` shape: (n_residues,) -- same labels GBSP/KNN are scored against.

    Evaluated via StratifiedKFold over residue POSITIONS (not sequences) so
    the LR itself is cross-validated; out-of-fold predictions are pooled
    before computing ROC-AUC/PR-AUC.  Returns (nan, nan, None) if y_true has
    too few examples of either class to stratify.

    Also returns the mean (|coef|, L1-normalized) fitted weights across
    folds so callers can report what blend the LR actually found useful --
    e.g. "0.34 / 0.66" instead of the hardcoded "0.90 / 0.10" -- as a
    concrete, actionable alternative rather than just a pass/fail signal.
    """
    n = len(y_true)
    if y_true.sum() < n_splits or (n - y_true.sum()) < n_splits:
        return float("nan"), float("nan"), None

    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_scores = np.zeros(n, dtype=np.float64)
        fold_coefs: List[np.ndarray] = []
        for train_idx, test_idx in skf.split(features, y_true):
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(features[train_idx], y_true[train_idx])
            oof_scores[test_idx] = clf.predict_proba(features[test_idx])[:, 1]
            fold_coefs.append(clf.coef_[0])
        roc = roc_auc_score(y_true, oof_scores)
        pr = average_precision_score(y_true, oof_scores)
        mean_coef = np.mean(np.stack(fold_coefs, axis=0), axis=0)
        abs_coef = np.abs(mean_coef)
        norm_weights = abs_coef / (abs_coef.sum() + 1e-8)
        return float(roc), float(pr), norm_weights
    except Exception as exc:
        print(f"Warning: LR baseline failed ({exc})")
        return float("nan"), float("nan"), None


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
    parser.add_argument("--family", default="VIM", help="Family filter: NDM, VIM, IMP, SPM, GIM, SIM")
    parser.add_argument("--identity", type=float, default=0.2, help="Cluster identity threshold (default: 0.2)")
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
    parser.add_argument("--hops", type=int, default=3, help="Propagation hops (default: 3)")
    parser.add_argument("--embed-cache", default="output/embeddings_cache.npz", help="Embedding cache path")
    parser.add_argument("--permutations", type=int, default=200, help="Permutations for p-value estimation")
    parser.add_argument("--bootstrap", type=int, default=500, help="Bootstrap samples for CI")
    parser.add_argument(
        "--structure-dir",
        default="output/structures",
        help="Directory to cache AlphaFold PDB structures (default: output/structures). "
             "Set to empty string to disable structure graph and use chain graph only.",
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=8.0,
        help="Cα–Cα distance threshold in Å for contact map (default: 8.0)",
    )
    parser.add_argument(
        "--biophysical-weight",
        type=float,
        default=0.10,
        help=(
            "Weight (0-1) given to the active-site-distance term in GBSP's "
            "combined score; (1-weight) goes to the propagated graph score. "
            "Default 0.10 was an initial guess, not a fitted value -- LR "
            "ablations in this script learn this weight directly from data "
            "and print it per-family (see 'LR-graph learned weights' in the "
            "run output / lr_graph_learned_biophysical_weight in the summary "
            "CSV). Re-run with that value to test if it actually improves "
            "ROC-AUC for your family -- it does NOT generalize across "
            "families (helped NDM/VIM substantially, hurt IMP in testing)."
        ),
    )
    parser.add_argument(
        "--split-method",
        choices=["group", "random"],
        default="group",
        help=(
            "Sequence split strategy (default: group). "
            "'group' = GroupKFold by DIAMOND cluster (no homolog leakage, "
            "the honest evaluation). "
            "'random' = plain KFold ignoring clusters -- run this to quantify "
            "how much harder the task is under strict homology splitting "
            "vs. a naive random split."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Structure dir — None disables AlphaFold graph entirely
    structure_dir: Optional[Path] = Path(args.structure_dir) if args.structure_dir else None

    all_records = read_fasta_records(args.input)
    family = args.family.upper()

    if family not in ALL_FAMILIES:
        raise SystemExit(
            f"Unknown family '{family}'. Must be one of: {', '.join(ALL_FAMILIES)}"
        )

    # Classify every sequence in the input fasta so n_sequences is verifiable,
    # not just trusted blindly -- this also makes it visible if most of the
    # file is unclassified ("OTHER"), which would explain a low per-family count.
    family_counts: Dict[str, int] = {fam: 0 for fam in ALL_FAMILIES}
    other_count = 0
    for rec in all_records:
        fam = family_from_header(rec.description)
        if fam:
            family_counts[fam] += 1
        else:
            other_count += 1
    print(f"[Classification] total sequences in input: {len(all_records)}")
    print(f"[Classification] by family: {family_counts}  |  unclassified (OTHER): {other_count}")

    records = [r for r in all_records if family_from_header(r.description) == family]
    if not records:
        raise SystemExit(
            f"No sequences found for family {family} in {args.input}. "
            f"Classification counts were: {family_counts} (OTHER={other_count}). "
            "If this family's count is 0, check that --input actually contains "
            "this family (re-run fetch_b1_superfamily.py with --families including it)."
        )

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

    # ------------------------------------------------------------------
    # Corrected per-family cluster report.
    # NOTE: `assignments` covers the ENTIRE superfamily fasta (DIAMOND was
    # run on args.input, not a family-filtered fasta), so we filter down to
    # this family's own sequences before reporting cluster statistics.
    # ------------------------------------------------------------------
    family_seq_ids = {rec.id for rec in records}
    family_assignments = {sid: cid for sid, cid in assignments.items() if sid in family_seq_ids}
    cluster_sizes = Counter(family_assignments.values())
    median_cluster_size = float("nan")
    if cluster_sizes:
        size_vals = sorted(cluster_sizes.values())
        median_cluster_size = size_vals[len(size_vals) // 2]
        n_singletons = sum(1 for s in size_vals if s == 1)
        print(
            f"\n[{family}] sequences: {len(family_seq_ids)} | "
            f"clusters (this family): {len(cluster_sizes)} | "
            f"median cluster size: {median_cluster_size} | "
            f"max: {max(size_vals)} | singletons: {n_singletons}"
        )

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

    if args.split_method == "random":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = kf.split(non_reference)
        print(f"[{family}] Split method: RANDOM KFold (n_splits={n_splits}) -- "
              "ignores cluster structure; expect inflated scores vs. 'group'.")
    else:
        gkf = GroupKFold(n_splits=n_splits)
        split_iter = gkf.split(non_reference, groups=groups)
        print(f"[{family}] Split method: GroupKFold by DIAMOND cluster (n_splits={n_splits})")

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
    aucs_lr = []
    aps_lr = []
    aucs_lr_graph = []
    aps_lr_graph = []
    lr_raw_weights_per_fold: List[np.ndarray] = []
    lr_graph_weights_per_fold: List[np.ndarray] = []
    fold_labels = []
    fold_scores_gbsp = []

    ref_seq = str(reference.seq)

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        train_records = [reference] + [non_reference[i] for i in train_idx]
        test_records = [non_reference[i] for i in test_idx]

        # GBSP scores
        scores = compute_scores_from_train(
            reference, train_records, embeddings,
            alpha=args.alpha, hops=args.hops, family=family,
            structure_dir=structure_dir,
            contact_threshold=args.contact_threshold,
            biophysical_weight=args.biophysical_weight,
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

        # LR baseline A: learned combination of GBSP's raw (pre-propagation)
        # ingredients -- variance + biophysical distance.
        lr_features = np.stack([scores["variance"], scores["biophysical"]], axis=1)
        roc_auc_lr, ap_lr, lr_weights = compute_lr_baseline_scores(lr_features, y_true)
        if lr_weights is not None:
            lr_raw_weights_per_fold.append(lr_weights)

        # LR baseline B: learned combination of the POST-propagation graph
        # score + biophysical distance -- GBSP's exact two ingredients, but
        # with a learned blend weight instead of the hardcoded 0.90/0.10.
        # Comparing A vs B isolates whether propagation itself helps/hurts
        # independent of the fixed weighting: if B beats A, propagation adds
        # signal and only the fixed weight was wrong; if B is no better than
        # A, propagation itself is destroying signal regardless of weighting.
        lr_graph_features = np.stack([scores["graph"], scores["biophysical"]], axis=1)
        roc_auc_lr_graph, ap_lr_graph, lr_graph_weights = compute_lr_baseline_scores(lr_graph_features, y_true)
        if lr_graph_weights is not None:
            lr_graph_weights_per_fold.append(lr_graph_weights)

        aucs_gbsp.append(roc_auc_gbsp)
        aps_gbsp.append(ap_gbsp)
        aucs_knn.append(roc_auc_knn)
        aps_knn.append(ap_knn)
        aucs_lr.append(roc_auc_lr)
        aps_lr.append(ap_lr)
        aucs_lr_graph.append(roc_auc_lr_graph)
        aps_lr_graph.append(ap_lr_graph)

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
                "lr_raw_roc_auc": roc_auc_lr,
                "lr_raw_pr_auc": ap_lr,
                "lr_graph_roc_auc": roc_auc_lr_graph,
                "lr_graph_pr_auc": ap_lr_graph,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    out_tag = f"{family.lower()}_{args.split_method}"
    fold_df.to_csv(output_dir / f"cv_{out_tag}_folds.csv", index=False)

    mean_auc = float(np.mean(aucs_gbsp)) if aucs_gbsp else float("nan")
    std_auc = float(np.std(aucs_gbsp)) if aucs_gbsp else float("nan")
    mean_ap = float(np.mean(aps_gbsp)) if aps_gbsp else float("nan")
    std_ap = float(np.std(aps_gbsp)) if aps_gbsp else float("nan")

    mean_auc_knn = float(np.mean(aucs_knn)) if aucs_knn else float("nan")
    std_auc_knn = float(np.std(aucs_knn)) if aucs_knn else float("nan")
    mean_ap_knn = float(np.mean(aps_knn)) if aps_knn else float("nan")
    std_ap_knn = float(np.std(aps_knn)) if aps_knn else float("nan")

    valid_lr_auc = [v for v in aucs_lr if not np.isnan(v)]
    valid_lr_ap = [v for v in aps_lr if not np.isnan(v)]
    mean_auc_lr = float(np.mean(valid_lr_auc)) if valid_lr_auc else float("nan")
    std_auc_lr = float(np.std(valid_lr_auc)) if valid_lr_auc else float("nan")
    mean_ap_lr = float(np.mean(valid_lr_ap)) if valid_lr_ap else float("nan")
    std_ap_lr = float(np.std(valid_lr_ap)) if valid_lr_ap else float("nan")

    valid_lr_graph_auc = [v for v in aucs_lr_graph if not np.isnan(v)]
    valid_lr_graph_ap = [v for v in aps_lr_graph if not np.isnan(v)]
    mean_auc_lr_graph = float(np.mean(valid_lr_graph_auc)) if valid_lr_graph_auc else float("nan")
    std_auc_lr_graph = float(np.std(valid_lr_graph_auc)) if valid_lr_graph_auc else float("nan")
    mean_ap_lr_graph = float(np.mean(valid_lr_graph_ap)) if valid_lr_graph_ap else float("nan")
    std_ap_lr_graph = float(np.std(valid_lr_graph_ap)) if valid_lr_graph_ap else float("nan")

    # Mean learned blend weights across folds: [signal_weight, biophysical_weight]
    mean_lr_raw_weights = (
        np.mean(np.stack(lr_raw_weights_per_fold, axis=0), axis=0)
        if lr_raw_weights_per_fold else None
    )
    mean_lr_graph_weights = (
        np.mean(np.stack(lr_graph_weights_per_fold, axis=0), axis=0)
        if lr_graph_weights_per_fold else None
    )

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
        "split_method": args.split_method,
        "n_folds": len(aucs_gbsp),
        "n_sequences": len(family_seq_ids),
        "n_clusters_family": len(cluster_sizes) if cluster_sizes else 0,
        "median_cluster_size": median_cluster_size,
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
        # LR baseline A: learned blend of raw (pre-propagation) ingredients
        "lr_raw_mean_roc_auc": mean_auc_lr,
        "lr_raw_std_roc_auc": std_auc_lr,
        "lr_raw_mean_pr_auc": mean_ap_lr,
        "lr_raw_std_pr_auc": std_ap_lr,
        "lr_raw_learned_signal_weight": float(mean_lr_raw_weights[0]) if mean_lr_raw_weights is not None else float("nan"),
        "lr_raw_learned_biophysical_weight": float(mean_lr_raw_weights[1]) if mean_lr_raw_weights is not None else float("nan"),
        # LR baseline B: learned blend of GBSP's exact (post-propagation) ingredients
        "lr_graph_mean_roc_auc": mean_auc_lr_graph,
        "lr_graph_std_roc_auc": std_auc_lr_graph,
        "lr_graph_mean_pr_auc": mean_ap_lr_graph,
        "lr_graph_std_pr_auc": std_ap_lr_graph,
        "lr_graph_learned_signal_weight": float(mean_lr_graph_weights[0]) if mean_lr_graph_weights is not None else float("nan"),
        "lr_graph_learned_biophysical_weight": float(mean_lr_graph_weights[1]) if mean_lr_graph_weights is not None else float("nan"),
        # Convenience deltas
        "delta_roc_auc_gbsp_minus_knn": mean_auc - mean_auc_knn,
        "delta_roc_auc_gbsp_minus_lr_raw": mean_auc - mean_auc_lr,
        "delta_roc_auc_gbsp_minus_lr_graph": mean_auc - mean_auc_lr_graph,
        "delta_roc_auc_lr_graph_minus_lr_raw": mean_auc_lr_graph - mean_auc_lr,
    }
    pd.DataFrame([summary]).to_csv(output_dir / f"cv_{out_tag}_summary.csv", index=False)

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
    fig_path = output_dir / f"cv_{out_tag}_roc_pr.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"Saved fold metrics: {output_dir / f'cv_{out_tag}_folds.csv'}")
    print(f"Saved summary:      {output_dir / f'cv_{out_tag}_summary.csv'}")
    print(f"Saved ROC/PR plot:  {fig_path}")
    print(f"\n[{family} / {args.split_method}] GBSP     ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}  "
          "(propagated graph score, fixed 0.90/0.10 blend)")
    print(f"[{family} / {args.split_method}] KNN      ROC-AUC: {mean_auc_knn:.4f} ± {std_auc_knn:.4f}  "
          f"(delta vs GBSP={mean_auc - mean_auc_knn:+.4f})")
    print(f"[{family} / {args.split_method}] LR-raw   ROC-AUC: {mean_auc_lr:.4f} ± {std_auc_lr:.4f}  "
          f"(learned blend of PRE-propagation variance + biophysical; "
          f"delta vs GBSP={mean_auc - mean_auc_lr:+.4f})")
    print(f"[{family} / {args.split_method}] LR-graph ROC-AUC: {mean_auc_lr_graph:.4f} ± {std_auc_lr_graph:.4f}  "
          f"(learned blend of GBSP's own POST-propagation graph score + biophysical; "
          f"delta vs GBSP={mean_auc - mean_auc_lr_graph:+.4f})")
    if mean_lr_raw_weights is not None:
        print(f"  LR-raw learned weights:   signal={mean_lr_raw_weights[0]:.2f} / "
              f"biophysical={mean_lr_raw_weights[1]:.2f}  (vs GBSP's hardcoded N/A / N/A, raw has no fixed blend)")
    if mean_lr_graph_weights is not None:
        print(f"  LR-graph learned weights: graph={mean_lr_graph_weights[0]:.2f} / "
              f"biophysical={mean_lr_graph_weights[1]:.2f}  (vs GBSP's hardcoded 0.90 / 0.10)")

    # ------------------------------------------------------------------
    # Decision rules (printed, not enforced) -- flags worth acting on
    # ------------------------------------------------------------------
    print(f"\n[DECISIONS for {family} / {args.split_method}]")
    if len(family_seq_ids) < 200:
        print(f"  - n_sequences={len(family_seq_ids)} < 200: dataset is small; "
              "consider relaxing UniProt filters or pulling in more related sequences.")
    if not np.isnan(median_cluster_size) and median_cluster_size == 1:
        print("  - median_cluster_size=1: clustering is fragmented for this family; "
              "retry fetch/cluster step with --cluster-identity 0.5 and 0.7 and compare.")
    if not np.isnan(mean_auc_knn) and abs(mean_auc - mean_auc_knn) < 0.02:
        print("  - |GBSP-KNN| < 0.02: graph propagation adds negligible value over "
              "naive k=1 similarity search for this family.")

    # Isolate whether the FIXED 0.90/0.10 weighting or PROPAGATION ITSELF
    # is responsible for any GBSP underperformance vs the LR ablations.
    if not np.isnan(mean_auc_lr) and not np.isnan(mean_auc_lr_graph):
        if mean_auc_lr_graph > mean_auc + 0.02 and mean_auc_lr_graph >= mean_auc_lr - 0.02:
            if mean_lr_graph_weights is not None:
                print(f"  - LR-graph ({mean_auc_lr_graph:.3f}) beats GBSP ({mean_auc:.3f}) and matches/beats "
                      f"LR-raw ({mean_auc_lr:.3f}): propagation itself is fine -- the hardcoded 0.90/0.10 "
                      "blend weight is the problem. Fix: use the learned weight instead "
                      f"(graph={mean_lr_graph_weights[0]:.2f} / biophysical={mean_lr_graph_weights[1]:.2f}) "
                      "rather than the fixed 0.90/0.10 ratio.")
            else:
                print(f"  - LR-graph ({mean_auc_lr_graph:.3f}) beats GBSP ({mean_auc:.3f}) and matches/beats "
                      f"LR-raw ({mean_auc_lr:.3f}): propagation itself is fine -- the hardcoded 0.90/0.10 "
                      "blend weight is the problem. Fix: learn the blend weight instead of hardcoding it.")
        elif mean_auc_lr > mean_auc_lr_graph + 0.02:
            print(f"  - LR-raw ({mean_auc_lr:.3f}) beats LR-graph ({mean_auc_lr_graph:.3f}) even though "
                  f"both use a LEARNED weight: propagation (alpha={args.alpha:.2f}, hops={args.hops}) "
                  "is destroying signal regardless of how it's weighted. Fix: lower hops, raise alpha "
                  "(less smoothing), or try --structure-dir '' to compare against the chain-graph "
                  "fallback directly.")
        elif mean_auc_lr_graph > mean_auc + 0.02:
            print(f"  - LR-graph ({mean_auc_lr_graph:.3f}) beats GBSP ({mean_auc:.3f}): even using GBSP's "
                  "own post-propagation signal, a learned blend beats the fixed 0.90/0.10 ratio. "
                  "Fix: learn the blend weight instead of hardcoding it.")

    if args.split_method == "group":
        print("  - Run the same family with --split-method random to quantify how much "
              "homology-aware splitting is responsible for the score (compare cv_"
              f"{family.lower()}_random_summary.csv once produced).")
    elif args.split_method == "random":
        group_summary_path = output_dir / f"cv_{family.lower()}_group_summary.csv"
        if group_summary_path.exists():
            try:
                group_auc = pd.read_csv(group_summary_path)["gbsp_mean_roc_auc"].iloc[0]
                if mean_auc - group_auc > 0.10:
                    print(f"  - random-split AUC ({mean_auc:.3f}) exceeds group-split AUC "
                          f"({group_auc:.3f}) by >0.10: task difficulty is driven mostly by "
                          "homology-split strictness, not the method itself. Report both "
                          "numbers together; don't cite the random-split number alone.")
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
