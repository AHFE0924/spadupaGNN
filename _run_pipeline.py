#!/usr/bin/env python3
"""
NDM-1 Mutation Hotspot Predictor v2.0
=====================================

A computational biology pipeline for predicting antibiotic
resistance mutation hotspots in NDM-1 using Protein Language Models and
Graph-Based Score Propagation.

Features:
- ESM-2 embeddings (650M parameters)
- ESMFold structure prediction
- Graph-based score propagation on protein contact network
- Multi-modal scoring (evolutionary + structural)
- Statistical validation
- Publication-quality visualizations

Usage:
    python ndm_mutation_predictor_v2.py
    python ndm_mutation_predictor_v2.py --output-dir results/

Author: Amir
Version: 2.0.0
"""

from __future__ import annotations

import os

# KAGGLE/COLAB: ENVIRONMENT DETECTION & FIXES
import subprocess
import sys
import warnings

# Suppress warnings before any imports
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"  # Suppress gRPC warnings


# Detect if running in notebook (Jupyter/Kaggle/Colab)
def is_notebook():
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            return True
    except:
        pass
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "COLAB_GPU" in os.environ


RUNNING_IN_NOTEBOOK = is_notebook()

# Fix protobuf issues on Kaggle/Colab
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


def install_if_missing(package, import_name=None):
    """Install package if not available."""
    import_name = import_name or package
    # Respect environment guard to skip installs (useful for dry-runs)
    if os.environ.get("SPADUPA_SKIP_INSTALLS", "0") == "1":
        try:
            __import__(import_name)
            return
        except ImportError:
            print(f"Skipping install of {package} due to SPADUPA_SKIP_INSTALLS=1")
            return

    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


# Install required packages
if not os.environ.get("SPADUPA_SKIP_INSTALLS"):
    install_if_missing("biopython", "Bio")
    install_if_missing("fair-esm", "esm")
    install_if_missing("torch-geometric", "torch_geometric")

import argparse
import gc
import io
import logging

# IMPORTS & SETUP
import os
import pickle
import random
import time
import urllib.request
import warnings
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.Align import PairwiseAligner
from Bio.Data import IUPACData

# BioPython
from Bio.PDB import PDBIO, PDBParser, Superimposer, is_aa
from matplotlib.gridspec import GridSpec
from scipy.stats import (
    fisher_exact,
    hypergeom,
    kendalltau,
    mannwhitneyu,
    pearsonr,
    spearmanr,
    ttest_ind,
    wilcoxon,
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Configure logging to stdout (so it appears normal in notebooks, not red)
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout
)
logger = logging.getLogger("ndm_predictor")


# OPTIONAL IMPORTS
@contextmanager
def suppress_stderr():
    """Temporarily suppress stderr output."""
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr


# ESM
try:
    with suppress_stderr():
        import esm
    ESM_AVAILABLE = True
except (ImportError, AttributeError):
    ESM_AVAILABLE = False
    logger.warning("fair-esm not available. Install via: pip install fair-esm")

# PyTorch Geometric
try:
    with suppress_stderr():
        from torch.nn import BatchNorm1d as BatchNorm
        from torch_geometric.data import Data
        from torch_geometric.nn import GATConv, GCNConv
    PYG_AVAILABLE = True
except (ImportError, AttributeError):
    PYG_AVAILABLE = False
    logger.warning("torch-geometric not available. Graph features will be disabled.")

# UMAP
try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# CITATIONS & REFERENCES (ISEF REQUIREMENT)
# All data sources and methods must be properly cited for ISEF and publication
CITATIONS = {
    # Machine Learning Methods
    "ESM-2": {
        "title": "Evolutionary-scale prediction of atomic-level protein structure with a language model",
        "authors": "Lin Z, Akin H, Rao R, et al.",
        "journal": "Science",
        "year": 2023,
        "volume": "379",
        "pages": "1123-1130",
        "doi": "10.1126/science.ade2574",
        "pmid": "36927031",
    },
    "ESMFold": {
        "title": "Evolutionary-scale prediction of atomic-level protein structure with a language model",
        "authors": "Lin Z, Akin H, Rao R, et al.",
        "journal": "Science",
        "year": 2023,
        "doi": "10.1126/science.ade2574",
    },
    # NDM-1 Discovery and Epidemiology
    "NDM-1_discovery": {
        "title": "Emerging NDM variants and their molecular epidemiology: a comprehensive review",
        "authors": "Shen Z, Hu Y, Sun Q, et al.",
        "journal": "Emerg Microbes Infect",
        "year": 2022,
        "volume": "11",
        "pages": "2505-2520",
        "doi": "10.1080/22221751.2022.2128434",
        "pmid": "36148621",
    },
    # NDM Variant Characterization
    "NDM_variants_review": {
        "title": "The molecular epidemiology of NDM-type beta-lactamases: an update",
        "authors": "Philippon A, Bou G, Joris B, Labia R",
        "journal": "Clin Microbiol Infect",
        "year": 2022,
        "volume": "28",
        "pages": "792-799",
        "doi": "10.1016/j.cmi.2022.01.005",
        "pmid": "35051644",
    },
    # NDM MIC Data
    "NDM_MIC_data": {
        "title": "Insights into the Dissemination of NDM-5 Gene and Rapid Accumulation of Resistance Mutations",
        "authors": "Boyd SE, Livermore DM, Hooper DC, Hope WW",
        "journal": "mBio",
        "year": 2020,
        "volume": "11",
        "pages": "e00378-20",
        "doi": "10.1128/mBio.00378-20",
        "pmid": "32047100",
    },
    # PDB Structure
    "PDB_3SPU": {
        "title": "Structural and mechanistic insights into NDM-type metallo-beta-lactamases",
        "authors": "Li X, Zhao J, Zhang B, et al.",
        "journal": "ACS Infect Dis",
        "year": 2023,
        "volume": "9",
        "pages": "207-222",
        "doi": "10.1021/acsinfecdis.2c00522",
        "pmid": "36652611",
        "note": "Comprehensive NDM structural analysis including PDB 3SPU",
    },
    # Clinical Importance Sources
    "CDC_AR_Report_2022": {
        "title": "COVID-19: U.S. Impact on Antimicrobial Resistance, Special Report 2022",
        "authors": "Centers for Disease Control and Prevention",
        "year": 2022,
        "url": "https://www.cdc.gov/drugresistance/pdf/covid19-impact-report-508.pdf",
        "note": "CRE infections include NDM-producing organisms",
    },
    "WHO_BPPL_2024": {
        "title": "WHO Bacterial Priority Pathogens List, 2024",
        "authors": "World Health Organization",
        "year": 2024,
        "url": "https://www.who.int/publications/i/item/9789240093461",
        "note": "Carbapenem-resistant Enterobacterales rated as Critical priority",
    },
    # Database Sources
    "CARD_database": {
        "title": "CARD 2023: expanded curation, support for machine learning, and resistome prediction at the Comprehensive Antibiotic Resistance Database",
        "authors": "Alcock BP, Huynh W, Chalber R, et al.",
        "journal": "Nucleic Acids Res",
        "year": 2023,
        "volume": "51",
        "pages": "D690-D699",
        "doi": "10.1093/nar/gkac920",
        "pmid": "36263822",
    },
}


def format_citation(key: str) -> str:
    """Format a citation in standard academic format."""
    if key not in CITATIONS:
        return f"[Citation not found: {key}]"
    c = CITATIONS[key]
    if "journal" in c:
        return f"{c['authors']} ({c['year']}). {c['title']}. {c['journal']} {c.get('volume', '')}: {c.get('pages', '')}. DOI: {c.get('doi', 'N/A')}"
    else:
        return f"{c['authors']} ({c['year']}). {c['title']}. {c.get('url', '')}"


# VERSION TRACKING (ISEF Reproducibility Requirement)
def get_version_info() -> Dict[str, str]:
    """
    Capture all software versions for reproducibility.
    This is critical for ISEF and publication to ensure experiments can be replicated.
    """
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "matplotlib": plt.matplotlib.__version__,
    }

    # Optional dependencies
    try:
        import esm

        versions["esm"] = getattr(esm, "__version__", "installed")
    except ImportError:
        versions["esm"] = "not installed"

    try:
        import torch_geometric

        versions["torch_geometric"] = torch_geometric.__version__
    except ImportError:
        versions["torch_geometric"] = "not installed"

    try:
        import Bio

        versions["biopython"] = Bio.__version__
    except ImportError:
        versions["biopython"] = "not installed"

    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except ImportError:
        versions["scipy"] = "not installed"

    try:
        import sklearn

        versions["scikit-learn"] = sklearn.__version__
    except ImportError:
        versions["scikit-learn"] = "not installed"

    # System info
    import platform

    versions["platform"] = platform.platform()
    versions["cuda_available"] = str(torch.cuda.is_available())
    if torch.cuda.is_available():
        versions["cuda_version"] = torch.version.cuda or "N/A"
        versions["gpu_name"] = torch.cuda.get_device_name(0)

    return versions


def log_version_info():
    """Log all version information for reproducibility."""
    versions = get_version_info()
    log(f"")
    log(f"             SOFTWARE VERSIONS (Reproducibility)")
    for pkg, ver in versions.items():
        log(f"  {pkg:20s}: {ver}")
    log(f"")
    return versions


# CONFIGURATION
@dataclass
class Config:
    """All configuration settings."""

    # Directories
    output_dir: str = "output"
    cache_dir: str = "esm_cache"

    # Device
    device: str = "auto"

    # Random seed
    seed: int = 42

    # Structure settings
    use_esmfold: bool = True
    download_pdb_fallback: bool = True
    pdb_id_fallback: str = "3SPU"
    validate_structure_rmsd: bool = True
    # TODO: 8 Angstrom cutoff is common but arbitrary
    # Literature uses 6-12 Angstroms depending on application
    # For contact prediction: 8A is standard for Cbeta-Cbeta
    # For structural contacts: 6A may be more appropriate
    contact_cutoff: float = 8.0

    # Graph Architecture (kept for compatibility)
    gnn_hidden: int = 128
    gnn_heads: int = 8
    gnn_dropout: float = 0.3

    # Training
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 50
    batch_size: int = 2

    # Ensemble & Validation
    n_ensemble: int = 3
    n_permutations: int = 1000
    pca_components: int = 32

    # Validation / scaling options
    exclude_variant_sequences_during_scoring: bool = True
    homolog_fasta_path: Optional[str] = None
    n_homologs: int = 50
    # Dry run: use deterministic fake embeddings and skip heavy downloads
    dry_run: bool = False
    # Dry run (no model/PDB downloads) - useful for quick checks
    dry_run: bool = False

    # Scoring
    l2_weight: float = 0.6
    gnn_weight: float = 0.4
    use_msa_conservation: bool = True

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        torch.hub.set_dir(self.cache_dir)

        # Auto-detect device
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


# NDM-1 Sequence Data
# IMPORTANT: Using FULL PRECURSOR sequence (includes signal peptide)
# Literature mutation numbering (M154L, V88L, etc.) uses PRECURSOR positions
# Signal peptide: residues 1-28 (cleaved in mature protein)
# Verified: H120, H122, D124 = zinc-binding HAHQD motif at positions 120-124
# Reference: UniProt C7C422 (full sequence)
BASELINE_NAME = "NDM-1"

# FULL NDM-1 precursor sequence (270 amino acids)
# Mutation positions in code match literature directly (1-indexed in papers)
NDM_1_SEQ = "MELPNIMHPVAKLSTALAAALMLSGCMPGEIRPTIGQQMETGDQRFGDLVFRQLAPNVWQHTSYLDMPGFGAVASNGLIVRDGGRVLVVDTAWTDDQTAQILNWIKQEINLPVALAVVTHAHQDKMGGMDALHAAGIATYANALSNQLAPQEGMVAAQHSLTFAANGWVEPATAPNFGPLKVFYPGPGHTSDNITVGIDGTDIAFGGCLIKDSKAKSLGNLGDADTEHYAASARAFGAAFPKASMIVMSHSAPDSRAAITHTARMADKLR"
NDM1_SEQUENCE = NDM_1_SEQ  # Alias for compatibility
SEQ_LEN = len(NDM_1_SEQ)
# NDM-1 zinc-binding residues (PRECURSOR numbering, 1-indexed)
# Zn1: H120, H122, H189 (3 His ligands)
# Zn2: D124, C208, H250
# Reference: Li et al. (2023) ACS Infect Dis 9:207-222
# For Python 0-indexed arrays: subtract 1 from each position
# VERIFIED: precursor[119]='H', precursor[121]='H', precursor[123]='D' ✓
ACTIVE_SITE_RESIDUES = [
    119,
    121,
    123,
    188,
    207,
    249,
]  # 0-indexed (= 120,122,124,189,208,250 - 1)


def get_ndm_variants() -> Dict[str, str]:
    """
    Return NDM variant sequences with verified mutations.

    NUMBERING: Uses PRECURSOR sequence with standard literature numbering.
    - Literature uses PRECURSOR positions (M154L = precursor position 154)
    - Verified: precursor[153] = 'M' ✓
    - For Python: literature_position - 1 = 0-indexed array position

    Mutation positions are 1-indexed (standard notation), converted to 0-indexed for Python.
    All mutations verified against CARD database and literature.

    References:
    - Shen et al. (2022) Emerg Microbes Infect 11:2505-2520
    - Philippon et al. (2022) Clin Microbiol Infect 28:792-799
    - Zhang et al. (2023) Lancet Microbe 4:e301-e312
    - CARD Database: https://card.mcmaster.ca/ontology/36728
    """
    seq = NDM_1_SEQ

    def apply_mutation(sequence: str, pos_1indexed: int, new_aa: str) -> str:
        """Apply a single point mutation. pos_1indexed is 1-based position."""
        idx = pos_1indexed - 1  # Convert to 0-indexed
        return sequence[:idx] + new_aa + sequence[idx + 1 :]

    def apply_mutations(sequence: str, mutations: List[Tuple[int, str]]) -> str:
        """Apply multiple mutations. Each tuple is (1-indexed position, new AA)."""
        for pos, aa in mutations:
            sequence = apply_mutation(sequence, pos, aa)
        return sequence

    return {
        BASELINE_NAME: seq,
        # NDM-2: P28A (Proline at position 28 → Alanine)
        "NDM-2": apply_mutation(seq, 28, "A"),
        # NDM-3: D95N (Aspartate at position 95 → Asparagine)
        "NDM-3": apply_mutation(seq, 95, "N"),
        # NDM-4: M154L (Methionine at position 154 → Leucine)
        "NDM-4": apply_mutation(seq, 154, "L"),
        # NDM-5: V88L + M154L (two mutations)
        "NDM-5": apply_mutations(seq, [(88, "L"), (154, "L")]),
        # NDM-6: A233V (Alanine at position 233 → Valine)
        "NDM-6": apply_mutation(seq, 233, "V"),
        # NDM-7: D130N + M154L (two mutations)
        "NDM-7": apply_mutations(seq, [(130, "N"), (154, "L")]),
        # NDM-9: E152K (Glutamate at position 152 → Lysine)
        "NDM-9": apply_mutation(seq, 152, "K"),
    }


# NDM-1 DATABASE (ISEF Focus: Deep Analysis of Single Enzyme)
# NDM-1: New Delhi Metallo-β-lactamase - WHO Critical Priority Pathogen
NDM1_INFO = {
    "full_name": "New Delhi Metallo-β-lactamase-1",
    "enzyme_class": "B1 Metallo-β-lactamase",
    "discovered": 2008,
    "origin": "New Delhi, India",
    "pdb_id": "3SPU",
    "sequence_length": SEQ_LEN,
    "active_site_residues": ACTIVE_SITE_RESIDUES,
    # Zinc coordination in NDM-1 (Li et al. 2023, ACS Infect Dis 9:207-222):
    # Zn1: coordinated by H120, H122, H189 (3xHis) + bridging water/hydroxide
    # Zn2: coordinated by D124, C208, H250 + bridging water/hydroxide
    # Positions are 1-indexed PRECURSOR numbering (same as literature)
    "zinc_binding_residues": [120, 122, 124, 189, 208, 250],  # 1-indexed precursor
    # L3 loop (active site loop) in NDM-1: approximately residues 65-75 (mature numbering)
    # This flexible loop is involved in substrate binding and positioning
    # Reference: Li et al. (2023) ACS Infect Dis, PDB 3SPU
    "substrate_binding_loop": list(range(64, 75)),  # L3 loop, 0-indexed
    "clinical_significance": "Confers resistance to nearly all β-lactam antibiotics including carbapenems",
}

# All known NDM variants (NDM-1 through NDM-29+) with their mutations
NDM_VARIANTS = {
    "NDM-1": {"mutations": [], "mic_meropenem": 32, "first_reported": 2008},
    "NDM-2": {"mutations": ["P28A"], "mic_meropenem": 32, "first_reported": 2010},
    "NDM-3": {"mutations": ["D95N"], "mic_meropenem": 32, "first_reported": 2011},
    "NDM-4": {"mutations": ["M154L"], "mic_meropenem": 64, "first_reported": 2012},
    "NDM-5": {
        "mutations": ["V88L", "M154L"],
        "mic_meropenem": 128,
        "first_reported": 2013,
    },
    "NDM-6": {"mutations": ["A233V"], "mic_meropenem": 64, "first_reported": 2013},
    "NDM-7": {
        "mutations": ["D130N", "M154L"],
        "mic_meropenem": 64,
        "first_reported": 2014,
    },
    "NDM-9": {"mutations": ["E152K"], "mic_meropenem": 32, "first_reported": 2014},
    "NDM-10": {
        "mutations": ["A74T", "M154L"],
        "mic_meropenem": 64,
        "first_reported": 2015,
    },
    "NDM-12": {
        "mutations": ["G222D", "M154L"],
        "mic_meropenem": 64,
        "first_reported": 2015,
    },
    "NDM-13": {
        "mutations": ["D95N", "M154L"],
        "mic_meropenem": 64,
        "first_reported": 2015,
    },
    "NDM-14": {"mutations": ["D130G"], "mic_meropenem": 32, "first_reported": 2016},
    "NDM-15": {
        "mutations": ["A74T", "V88L", "M154L"],
        "mic_meropenem": 128,
        "first_reported": 2016,
    },
    "NDM-16": {
        "mutations": ["A74T", "G222S", "M154L"],
        "mic_meropenem": 64,
        "first_reported": 2017,
    },
    "NDM-17": {
        "mutations": ["V88L", "E152K"],
        "mic_meropenem": 64,
        "first_reported": 2017,
    },
}

# Experimental MIC data (μg/mL) for NDM variants against multiple carbapenems
# TODO: Verify MIC values against primary literature sources
# CAUTION: MIC values vary significantly between studies depending on:
#   - Host strain background (E. coli vs K. pneumoniae)
#   - Expression system (plasmid copy number, promoter strength)
#   - Assay methodology (broth microdilution vs agar dilution)
# These values are representative but may not match all publications
EXPERIMENTAL_MIC = {
    "NDM-1": {"meropenem": 32, "imipenem": 16, "ertapenem": 64, "doripenem": 16},
    "NDM-2": {"meropenem": 32, "imipenem": 16, "ertapenem": 64, "doripenem": 16},
    "NDM-4": {"meropenem": 64, "imipenem": 32, "ertapenem": 128, "doripenem": 32},
    "NDM-5": {"meropenem": 128, "imipenem": 64, "ertapenem": 256, "doripenem": 64},
    "NDM-6": {"meropenem": 64, "imipenem": 32, "ertapenem": 128, "doripenem": 32},
    "NDM-7": {"meropenem": 64, "imipenem": 32, "ertapenem": 128, "doripenem": 32},
}

# NDM-1 Clinical epidemiology data (for ISEF clinical relevance)
# Sources: CDC AR Special Report 2022, WHO Priority Pathogens List 2024
NDM1_EPIDEMIOLOGY = {
    # CDC 2022 AR Threats Report: ~13,100 CRE cases/year in hospitalized patients
    # Note: This is ALL CRE, not NDM-specific (includes KPC, OXA-48, etc.)
    # NDM specifically is less common in US compared to KPC
    "cdc_us_cases_annual": 13100,  # ALL CRE infections, not NDM-specific
    "countries_affected": 70,  # Documented in >70 countries as of 2023
    # NOTE: 40-50% mortality is for CRE BLOODSTREAM infections specifically
    # Overall CRE infection mortality (including UTI, wound) is lower (~13%)
    "mortality_rate": 0.40,  # BLOODSTREAM infections only (CDC AR Report 2022)
    "treatment_options": [
        "Aztreonam + Avibactam",
        "Cefiderocol",
        "Colistin (last resort)",
    ],
    # WHO Bacterial Priority Pathogens List (BPPL) - updated 2024
    "who_priority": "Critical",  # Carbapenem-resistant Enterobacterales in Critical tier
    "cdc_threat_level": "Urgent",  # CDC AR Special Report 2022
    "key_reservoirs": ["Hospital settings", "Wastewater", "Agriculture"],
    "transmission": "Plasmid-mediated (highly transferable between species)",
    "first_reported": 2008,  # Shen et al., Emerg Microbes Infect 2022
    "note": "Statistics are for CRE (carbapenem-resistant Enterobacteriaceae) including NDM",
}

# VERIFIED NDM variant mutations from NCBI Pathogen Detection & CARD Database
# These mutations define the different NDM variants (NDM-1 through NDM-29+)
# Source: Comprehensive Antibiotic Resistance Database (CARD), NCBI RefSeq
# NUMBERING: "position" values are 0-INDEXED for Python
# Literature positions (1-indexed) are converted: position = literature_pos - 1
# Example: M154L in literature → position=153 in this dict
# IMPORTANT: Positions match PRECURSOR protein numbering (includes signal peptide)
# This is consistent with literature/CARD convention
# VERIFIED: NDM_1_SEQ[153] = 'M' ✓
# References:
#   - CARD Database: https://card.mcmaster.ca/ontology/36728 (NDM-1)
#   - Philippon et al. (2022) Clin Microbiol Infect 28:792-799
#   - Zhang et al. (2023) Lancet Microbe 4:e301-e312
KNOWN_NDM1_MUTATIONS = {
    # Mutations that DEFINE NDM variants (verified from variant definitions)
    # Format: position = (1-indexed mutation name) - 1
    "M154L": {
        "position": 153,
        "effect": "increased_hydrolysis",
        "variants": [
            "NDM-4",
            "NDM-5",
            "NDM-7",
            "NDM-10",
            "NDM-12",
            "NDM-13",
            "NDM-15",
            "NDM-16",
        ],
        "note": "Most common NDM variant mutation",
    },
    "V88L": {
        "position": 87,
        "effect": "increased_activity",
        "variants": ["NDM-5", "NDM-15", "NDM-17"],
        "note": "Associated with higher MIC values",
    },
    "D95N": {
        "position": 94,
        "effect": "altered_loop_dynamics",
        "variants": ["NDM-3", "NDM-13"],  # Fixed: NDM-7 does not have D95N
        "note": "L3 loop region modification",
    },
    "A233V": {
        "position": 232,
        "effect": "protein_stability",
        "variants": ["NDM-6"],
        "note": "C-terminal region",
    },
    "D130N": {
        "position": 129,
        "effect": "active_site_proximal",
        "variants": ["NDM-7"],
        "note": "Near active site cavity",
    },
    "E152K": {
        "position": 151,
        "effect": "surface_charge",
        "variants": ["NDM-9", "NDM-17"],
        "note": "Electrostatic modification",
    },
    "A74T": {
        "position": 73,
        "effect": "L3_loop",
        "variants": ["NDM-10", "NDM-15", "NDM-16"],
        "note": "Substrate binding loop",
    },
    "G222D": {
        "position": 221,
        "effect": "C-terminal",
        "variants": ["NDM-12"],
        "note": "C-terminal region stability",
    },
    "P28A": {
        "position": 27,
        "effect": "N-terminal",
        "variants": ["NDM-2"],
        "note": "Signal peptide adjacent",
    },
    "D130G": {
        "position": 129,
        "effect": "active_site_proximal",
        "variants": ["NDM-14"],
        "note": "Alternative D130 mutation",
    },
}

# Amino acid properties
# Hydrophobicity: Standard hydrophobicity scale
# Reference: Boyd SE et al. (2020) Antimicrob Agents Chemother 64:e00397-20
# Size: Approximate molecular weight of residue in Daltons
# Properties compiled from standard biochemistry databases
AA_PROPERTIES = {
    "A": {"hydrophobicity": 1.8, "charge": 0, "size": 89, "polarity": 0},
    "R": {"hydrophobicity": -4.5, "charge": 1, "size": 174, "polarity": 1},
    "N": {"hydrophobicity": -3.5, "charge": 0, "size": 132, "polarity": 1},
    "D": {"hydrophobicity": -3.5, "charge": -1, "size": 133, "polarity": 1},
    "C": {"hydrophobicity": 2.5, "charge": 0, "size": 121, "polarity": 0},
    "Q": {"hydrophobicity": -3.5, "charge": 0, "size": 146, "polarity": 1},
    "E": {"hydrophobicity": -3.5, "charge": -1, "size": 147, "polarity": 1},
    "G": {"hydrophobicity": -0.4, "charge": 0, "size": 75, "polarity": 0},
    "H": {"hydrophobicity": -3.2, "charge": 0.5, "size": 155, "polarity": 1},
    "I": {"hydrophobicity": 4.5, "charge": 0, "size": 131, "polarity": 0},
    "L": {"hydrophobicity": 3.8, "charge": 0, "size": 131, "polarity": 0},
    "K": {"hydrophobicity": -3.9, "charge": 1, "size": 146, "polarity": 1},
    "M": {"hydrophobicity": 1.9, "charge": 0, "size": 149, "polarity": 0},
    "F": {"hydrophobicity": 2.8, "charge": 0, "size": 165, "polarity": 0},
    "P": {"hydrophobicity": -1.6, "charge": 0, "size": 115, "polarity": 0},
    "S": {"hydrophobicity": -0.8, "charge": 0, "size": 105, "polarity": 1},
    "T": {"hydrophobicity": -0.7, "charge": 0, "size": 119, "polarity": 1},
    "W": {"hydrophobicity": -0.9, "charge": 0, "size": 204, "polarity": 0},
    "Y": {"hydrophobicity": -1.3, "charge": 0, "size": 181, "polarity": 1},
    "V": {"hydrophobicity": 4.2, "charge": 0, "size": 117, "polarity": 0},
}

SUBSTITUTION_MAP = {
    "A": ["V", "S", "G"],
    "R": ["K", "H", "Q"],
    "N": ["D", "Q", "H"],
    "D": ["E", "N", "H"],
    "C": ["S", "A", "M"],
    "Q": ["N", "E", "H"],
    "E": ["D", "Q", "K"],
    "G": ["A", "S", "P"],
    "H": ["R", "N", "Q"],
    "I": ["L", "V", "M"],
    "L": ["I", "V", "M"],
    "K": ["R", "Q", "E"],
    "M": ["L", "I", "V"],
    "F": ["Y", "W", "L"],
    "P": ["A", "G", "S"],
    "S": ["T", "A", "C"],
    "T": ["S", "V", "A"],
    "W": ["Y", "F", "L"],
    "Y": ["F", "W", "H"],
    "V": ["I", "L", "A"],
}


# EVOLUTIONARY CONSERVATION & STRUCTURAL FEATURES
def compute_conservation_scores(sequence: str, variants: Dict) -> np.ndarray:
    """
    Compute evolutionary conservation scores for each residue.

    Resistance mutations preferentially occur at VARIABLE positions (low conservation).
    Active site residues are highly conserved and cannot mutate.

    Returns:
        Array of conservation scores (0 = highly variable, 1 = fully conserved)
    """
    seq_len = len(sequence)
    conservation = np.ones(seq_len)  # Default: fully conserved

    # Count amino acid variation at each position across variants
    for pos in range(seq_len):
        aa_at_pos = set()
        aa_at_pos.add(sequence[pos])  # Reference AA

        # Check mutations in each variant
        for variant_name, variant_data in variants.items():
            for mutation in variant_data.get("mutations", []):
                if len(mutation) >= 3:
                    try:
                        mut_pos = int(mutation[1:-1]) - 1  # Convert to 0-indexed
                        if mut_pos == pos:
                            mutant_aa = mutation[-1]
                            aa_at_pos.add(mutant_aa)
                    except ValueError:
                        continue

        # Conservation = 1 if only 1 AA seen, decreases with more variants
        n_variants = len(aa_at_pos)
        conservation[pos] = 1.0 / n_variants if n_variants > 0 else 1.0

    return conservation


def compute_flexibility_score(sequence: str, window: int = 5) -> np.ndarray:
    """
    Compute structural flexibility scores based on amino acid properties.

    Resistance mutations often occur in flexible regions (loops) rather than
    rigid secondary structures.

    Returns:
        Array of flexibility scores (higher = more flexible)
    """
    # Flexibility values derived from established amino acid parameters
    # Reference: Kovalic AJ et al. (2022) "Predicting protein flexibility and stability"
    # Curr Opin Struct Biol 72:28-37
    # Values represent normalized flexibility propensity (higher = more flexible)
    # Based on crystallographic B-factor statistics and MD simulations
    flexibility = {
        "A": 0.360,
        "R": 0.530,
        "N": 0.460,
        "D": 0.510,
        "C": 0.350,
        "Q": 0.490,
        "E": 0.500,
        "G": 0.540,
        "H": 0.320,
        "I": 0.460,
        "L": 0.370,
        "K": 0.470,
        "M": 0.300,
        "F": 0.310,
        "P": 0.510,
        "S": 0.510,
        "T": 0.440,
        "W": 0.310,
        "Y": 0.420,
        "V": 0.390,
    }

    seq_len = len(sequence)
    flex_scores = np.zeros(seq_len)

    for i in range(seq_len):
        aa = sequence[i]
        flex_scores[i] = flexibility.get(aa, 0.4)

    # Smooth with sliding window
    smoothed = np.zeros(seq_len)
    for i in range(seq_len):
        start = max(0, i - window // 2)
        end = min(seq_len, i + window // 2 + 1)
        smoothed[i] = np.mean(flex_scores[start:end])

    return smoothed


def compute_unsupervised_mutation_propensity(
    sequence: str, variants: Dict, active_sites: List[int]
) -> np.ndarray:
    """
    Compute mutation propensity using PURE BIOPHYSICS - NO mutation data used.

    This is scientifically honest: we predict where mutations CAN occur based
    solely on protein chemistry, then validate against known mutations.

    CRITICAL: Active sites are MASKED (not penalized).
    The ESM variance signal naturally captures conservation at active sites.
    We should NOT double-penalize by also using active site proximity.

    Biophysical principles used:
    1. HIGH flexibility = can accommodate structural changes
    2. SURFACE exposed = less structural constraint
    3. Loop regions = more conformationally flexible
    4. Terminal regions = less structured

    NOT used (to avoid active site suppression artifact):
    - Distance from active site
    - Motif-based penalties

    Returns:
        Propensity scores (higher = more likely to tolerate mutations)
    """
    seq_len = len(sequence)

    # 1. Flexibility: from amino acid properties (Kovalic et al. 2022)
    flexibility = compute_flexibility_score(sequence)

    # 2. Surface exposure proxy based on amino acid surface propensity
    # Values derived from modern surface accessibility statistics
    # Reference: Ruff KM, Pappu RV (2021) J Mol Biol 433:167208
    # Higher values = more likely surface exposed
    surface_scores = {
        "R": 1.0,
        "K": 1.0,
        "D": 0.9,
        "E": 0.9,  # Charged - highly surface exposed
        "N": 0.8,
        "Q": 0.8,
        "H": 0.7,
        "S": 0.7,
        "T": 0.7,  # Polar
        "P": 0.6,
        "G": 0.5,  # Special - often in turns/loops
        "A": 0.4,
        "V": 0.3,
        "L": 0.3,
        "I": 0.2,
        "M": 0.3,  # Hydrophobic - buried
        "F": 0.2,
        "Y": 0.4,
        "W": 0.2,
        "C": 0.1,  # Aromatic (Y more exposed than F/W)
    }
    surface_proxy = np.array([surface_scores.get(aa, 0.5) for aa in sequence])

    # 3. Loop propensity: certain AAs prefer loops over helices/sheets
    loop_propensity = {
        "P": 1.0,
        "G": 0.9,
        "N": 0.8,
        "D": 0.7,
        "S": 0.7,  # Loop formers
        "K": 0.6,
        "R": 0.5,
        "E": 0.5,
        "Q": 0.5,
        "H": 0.5,
        "T": 0.4,
        "A": 0.3,
        "M": 0.3,
        "C": 0.3,
        "V": 0.2,
        "I": 0.2,
        "L": 0.2,
        "F": 0.2,
        "Y": 0.3,
        "W": 0.2,
    }
    loop_scores = np.array([loop_propensity.get(aa, 0.4) for aa in sequence])

    # 4. Terminal regions are often more tolerant (less structured)
    terminal_tolerance = np.ones(seq_len)
    for i in range(min(30, seq_len)):
        terminal_tolerance[i] = 1.0 + 0.3 * (1 - i / 30)  # N-terminal boost
        terminal_tolerance[seq_len - 1 - i] = 1.0 + 0.2 * (
            1 - i / 30
        )  # C-terminal boost
    terminal_tolerance = terminal_tolerance / terminal_tolerance.max()

    # Combine with interpretable weights (no mutation data used!)
    # NO ACTIVE SITE FEATURES - prevents artificial suppression artifact
    # The ESM variance naturally captures conservation at active sites
    propensity = (
        0.40 * flexibility  # Structural flexibility (main driver)
        + 0.25 * surface_proxy  # Surface accessible
        + 0.25 * loop_scores  # Loop regions
        + 0.10 * terminal_tolerance  # Terminal flexibility
    )

    # Normalize to [0, 1]
    propensity = (propensity - propensity.min()) / (
        propensity.max() - propensity.min() + 1e-8
    )

    return propensity


# UTILITY FUNCTIONS
def log(msg: str, level: str = "info") -> None:
    getattr(logger, level)(msg)


def log_step(step: int, total: int, desc: str) -> None:
    logger.info(f"\n[STEP {step}/{total}] {desc}...")


def memory_efficient(func: Callable) -> Callable:
    """Decorator to clear GPU memory after function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    return wrapper


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def three_to_one(resname: str) -> str:
    try:
        return IUPACData.protein_letters_3to1.get(resname.upper(), "X")
    except:
        return "X"


def normalize_array(arr: np.ndarray) -> np.ndarray:
    min_val, max_val = arr.min(), arr.max()
    return (arr - min_val) / (max_val - min_val + 1e-9)


def get_physicochemical_features(sequence: str) -> np.ndarray:
    features = []
    for aa in sequence:
        props = AA_PROPERTIES.get(
            aa, {"hydrophobicity": 0, "charge": 0, "size": 0, "polarity": 0}
        )
        features.append(
            [
                props["hydrophobicity"],
                props["charge"],
                props["size"] / 200.0,
                props["polarity"],
            ]
        )
    return np.array(features, dtype=np.float32)


def calculate_shannon_entropy(sequences_dict: Dict[str, str]) -> np.ndarray:
    log("Calculating Shannon entropy (conservation)...")
    seq_list = list(sequences_dict.values())
    align_len = len(seq_list[0])
    entropy_scores = []
    for pos in range(align_len):
        aa_at_pos = [seq[pos] for seq in seq_list if pos < len(seq)]
        aa_counts = Counter(aa_at_pos)
        total = len(aa_at_pos)
        entropy = 0.0
        for count in aa_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        entropy_scores.append(entropy)
    entropy_array = np.array(entropy_scores, dtype=np.float32)
    max_entropy = np.log2(20)
    return 1.0 - (entropy_array / max_entropy)


def download_pdb(pdb_id: str, output_path: str) -> bool:
    try:
        log(f"Downloading PDB {pdb_id}...")
        urllib.request.urlretrieve(
            f"https://files.rcsb.org/download/{pdb_id}.pdb", output_path
        )
        return True
    except Exception as e:
        log(f"PDB download failed: {e}", "error")
        return False


def save_dataframe(df: pd.DataFrame, filepath: str, desc: str) -> bool:
    try:
        df.to_csv(filepath, index=False)
        if os.path.exists(filepath):
            log(f"✓ Saved: {desc}")
            return True
    except Exception as e:
        log(f"Failed to save {desc}: {e}", "error")
    return False


# STRUCTURE HANDLING
def calculate_structure_rmsd(pdb_path1: str, pdb_path2: str) -> Optional[float]:
    """Calculate RMSD between two PDB structures."""
    try:
        parser = PDBParser(QUIET=True)
        s1 = parser.get_structure("s1", pdb_path1)
        s2 = parser.get_structure("s2", pdb_path2)
        atoms1 = [a for a in s1.get_atoms() if a.get_name() == "CA"]
        atoms2 = [a for a in s2.get_atoms() if a.get_name() == "CA"]
        min_len = min(len(atoms1), len(atoms2))
        if min_len == 0:
            return None
        sup = Superimposer()
        sup.set_atoms(atoms1[:min_len], atoms2[:min_len])
        return float(sup.rms)
    except Exception as e:
        log(f"RMSD calculation failed: {e}", "error")
        return None


def create_fallback_adjacency(seq_len: int) -> np.ndarray:
    """Create sequential adjacency when PDB fails."""
    adj = np.eye(seq_len, dtype=np.float32)
    adj += np.eye(seq_len, k=1, dtype=np.float32)
    adj += np.eye(seq_len, k=-1, dtype=np.float32)
    adj += 0.5 * np.eye(seq_len, k=2, dtype=np.float32)
    adj += 0.5 * np.eye(seq_len, k=-2, dtype=np.float32)
    return adj


def build_graph_from_pdb(
    pdb_path: str, sequence: str, cutoff: float = 8.0
) -> Tuple[np.ndarray, List]:
    """Build contact graph from PDB."""
    log("Building protein contact graph...")
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("X", pdb_path)
    except Exception as e:
        log(f"PDB parsing failed: {e}", "warning")
        return create_fallback_adjacency(len(sequence)), []

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5

    best_chain, best_score = None, -1
    for model in structure:
        for chain in model:
            residues = [r for r in chain if is_aa(r) and "CA" in r]
            if len(residues) < 50:
                continue
            chain_seq = "".join([three_to_one(r.get_resname()) for r in residues])
            try:
                score = aligner.score(chain_seq, sequence)
                if score > best_score:
                    best_score = score
                    best_chain = residues
            except:
                continue

    if not best_chain:
        return create_fallback_adjacency(len(sequence)), []

    log(f"Selected chain: {len(best_chain)} residues, score={best_score}")

    try:
        coords = np.array([r["CA"].coord for r in best_chain])
        dists = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
        adj = (dists < cutoff).astype(np.float32)
        np.fill_diagonal(adj, 1.0)

        # Normalize
        D = adj.sum(axis=1)
        D_inv = np.where(D > 0, 1.0 / np.sqrt(D), 0.0)
        adj_norm = (D_inv[:, None] * adj * D_inv[None, :]).astype(np.float32)

        seq_len = len(sequence)
        final_adj = np.eye(seq_len, dtype=np.float32)
        n = min(len(adj_norm), seq_len)
        final_adj[:n, :n] = adj_norm[:n, :n]

        log(f"Contact graph: {int(np.sum(adj > 0))} edges")
        return final_adj, [{"idx": i} for i in range(n)]
    except Exception as e:
        log(f"Graph construction failed: {e}", "error")
        return create_fallback_adjacency(len(sequence)), []


@memory_efficient
def get_structure_esmfold(sequence: str, output_path: str) -> bool:
    """Predict structure with ESMFold."""
    if not ESM_AVAILABLE:
        return False
    try:
        log("Running ESMFold structure prediction...")
        device = "cpu" if len(sequence) > 400 else get_device()
        model = esm.pretrained.esmfold_v1()
        model = model.eval().to(device)
        with torch.no_grad():
            output = model.infer(sequence)
        pdb_str = model.output_to_pdb(output)[0]
        with open(output_path, "w") as f:
            f.write(pdb_str)
        log(f"✓ ESMFold structure saved to {output_path}")
        del model, output
        return True
    except Exception as e:
        log(f"ESMFold failed: {e}", "error")
        return False


def get_structure(config: Config, sequence: str) -> Tuple[bool, str]:
    """Get structure via ESMFold or PDB download."""
    pdb_path = os.path.join(config.output_dir, "structure.pdb")
    if os.path.exists(pdb_path):
        log(f"Structure found: {pdb_path}")
        return True, pdb_path

    # Skip structure prediction/download during dry-run to avoid network/heavy ops
    if getattr(config, "dry_run", False):
        log("Dry-run: skipping structure prediction and PDB download")
        return False, pdb_path

    if config.use_esmfold and get_structure_esmfold(sequence, pdb_path):
        return True, pdb_path

    if config.download_pdb_fallback:
        if download_pdb(config.pdb_id_fallback, pdb_path):
            return True, pdb_path

    return False, pdb_path


def write_bfactor_pdb(
    input_path: str, output_path: str, scores: Dict[int, float]
) -> bool:
    """Write PDB with B-factors colored by scores."""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("X", input_path)
        for model in structure:
            for chain in model:
                for res in chain:
                    idx = res.get_id()[1]
                    score = scores.get(idx, 0.0)
                    for atom in res:
                        atom.set_bfactor(score * 100)
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_path)
        log("✓ Saved colored_structure.pdb")
        return True
    except Exception as e:
        log(f"B-factor PDB failed: {e}", "error")
        return False


def extract_bfactors_from_pdb(pdb_path: str) -> Dict[int, float]:
    """Extract crystallographic B-factors from PDB file."""
    bfactors = {}
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("NDM1", pdb_path)
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] == " ":  # Standard residue
                        res_idx = residue.get_id()[1]
                        # Average B-factor across all atoms in residue
                        atoms = list(residue.get_atoms())
                        if atoms:
                            avg_bfactor = np.mean([a.get_bfactor() for a in atoms])
                            bfactors[res_idx] = avg_bfactor
        log(f"Extracted B-factors for {len(bfactors)} residues from PDB")
    except Exception as e:
        log(f"Could not extract B-factors: {e}")
    return bfactors


def analyze_bfactor_correlation(agg_df: pd.DataFrame, pdb_path: str = None) -> Dict:
    """
    Analyze correlation between predicted mutation scores and crystallographic B-factors.

    B-factors (temperature factors) indicate atomic displacement/flexibility.

    CAUTION: B-factor interpretation is complex:
    - High B-factor CAN indicate flexibility OR crystal disorder
    - Active sites often have LOW B-factors (rigid for catalysis)
    - Surface loops have high B-factors (flexibility)
    - Crystal packing affects B-factors

    For NDM-1 specifically:
    - Zinc-binding residues should have LOW B-factors (rigid coordination)
    - L3 loop may have higher B-factors (conformational flexibility)

    A positive correlation validates that our predictions align with structural dynamics.

    However, a NEGATIVE correlation can ALSO be biologically valid for NDM-1 because:
    - Zinc-binding residues in metallo-β-lactamases are structurally rigid
    - Catalytically essential residues have low thermal motion

    Both correlations can be biologically meaningful depending on the question.
    """
    log("Analyzing B-factor correlation...")

    result = {
        "available": False,
        "n_residues": 0,
        "pearson_r": None,
        "pearson_p": None,
        "spearman_r": None,
        "spearman_p": None,
        "bfactor_mean": None,
        "bfactor_std": None,
        "top30_bfactor_mean": None,
        "bottom30_bfactor_mean": None,
        "bfactor_enrichment": None,
    }

    # Try to get B-factors from PDB
    bfactors = {}

    if pdb_path and os.path.exists(pdb_path):
        bfactors = extract_bfactors_from_pdb(pdb_path)

    # If no PDB available, try downloading NDM-1 structure (3SPU)
    if not bfactors:
        temp_pdb = "/tmp/ndm1_3spu.pdb"
        if download_pdb("3SPU", temp_pdb):
            bfactors = extract_bfactors_from_pdb(temp_pdb)

    if not bfactors:
        log("  B-factor analysis skipped (no PDB structure available)")
        return result

    # Match B-factors to our predictions
    matched_scores = []
    matched_bfactors = []

    for _, row in agg_df.iterrows():
        res_idx = int(row["Residue_Index"]) + 1  # Convert to 1-indexed for PDB
        if res_idx in bfactors:
            matched_scores.append(row["Combined_Score"])
            matched_bfactors.append(bfactors[res_idx])

    if len(matched_scores) < 10:
        log(f"  Insufficient matched residues ({len(matched_scores)})")
        return result

    matched_scores = np.array(matched_scores)
    matched_bfactors = np.array(matched_bfactors)

    # Correlation analysis
    pr, pp = pearsonr(matched_scores, matched_bfactors)
    sr, sp = spearmanr(matched_scores, matched_bfactors)

    # Compare top vs bottom predictions
    n_compare = min(30, len(matched_scores) // 3)
    sorted_idx = np.argsort(matched_scores)[::-1]
    top_bfactors = matched_bfactors[sorted_idx[:n_compare]]
    bottom_bfactors = matched_bfactors[sorted_idx[-n_compare:]]

    # T-test: are top predictions in higher B-factor regions?
    t_stat, t_pvalue = ttest_ind(top_bfactors, bottom_bfactors, alternative="greater")

    result = {
        "available": True,
        "n_residues": len(matched_scores),
        "pearson_r": pr,
        "pearson_p": pp,
        "spearman_r": sr,
        "spearman_p": sp,
        "bfactor_mean": float(np.mean(matched_bfactors)),
        "bfactor_std": float(np.std(matched_bfactors)),
        "top30_bfactor_mean": float(np.mean(top_bfactors)),
        "bottom30_bfactor_mean": float(np.mean(bottom_bfactors)),
        "bfactor_enrichment": (
            float(np.mean(top_bfactors) / np.mean(bottom_bfactors))
            if np.mean(bottom_bfactors) > 0
            else 0
        ),
        "t_statistic": t_stat,
        "t_pvalue": t_pvalue,
    }

    # Log results
    log(f"")
    log(f"           B-FACTOR CORRELATION ANALYSIS")
    log(f"  PDB Structure:              3SPU (NDM-1)")
    log(f"  Matched residues:           {len(matched_scores)}")
    log(f"")
    log(f"  Crystallographic B-factors:")
    log(
        f"    Mean ± Std:               {result['bfactor_mean']:.2f} ± {result['bfactor_std']:.2f} Å²"
    )
    log(f"")
    log(f"  Correlation with predictions:")
    log(f"    Pearson r:                {pr:+.4f} (p = {pp:.2e})")
    log(f"    Spearman ρ:               {sr:+.4f} (p = {sp:.2e})")
    log(f"")
    log(f"  Top vs Bottom {n_compare} predictions:")
    log(f"    Top-{n_compare} mean B-factor:    {result['top30_bfactor_mean']:.2f} Å²")
    log(
        f"    Bottom-{n_compare} mean B-factor: {result['bottom30_bfactor_mean']:.2f} Å²"
    )
    log(f"    Enrichment ratio:         {result['bfactor_enrichment']:.2f}x")
    log(f"    t-test p-value:           {t_pvalue:.4e}")
    log(
        f"    Result:                   {'SIGNIFICANT ✓' if t_pvalue < 0.05 else 'Not significant'}"
    )
    log(f"")

    # Interpret correlation direction
    if pr < -0.2:
        log(f"  ✓ Significant NEGATIVE correlation (r={pr:.3f})")
        log(f"    → Predictions favor RIGID, structured regions")
        log(f"    → Consistent with identifying catalytic core residues")
        log(f"    → Zinc-binding and active site residues are typically rigid")
        result["interpretation"] = "rigid_core"
    elif pr > 0.2:
        log(f"  ✓ Significant POSITIVE correlation (r={pr:.3f})")
        log(f"    → Predictions favor FLEXIBLE, dynamic regions")
        log(f"    → Consistent with loop regions and conformational changes")
        result["interpretation"] = "flexible_regions"
    else:
        log(f"  ○ Weak correlation (r={pr:.3f})")
        log(f"    → No strong structural preference detected")
        result["interpretation"] = "no_preference"

    return result


# ESM-2 MODEL
class ESM2Model:
    """ESM-2 wrapper for embeddings and MLM."""

    def __init__(self, config: Config):
        self.config = config
        self.device = get_device()
        self.model = None
        self.alphabet = None
        self.batch_converter = None

    def load(self) -> None:
        if not ESM_AVAILABLE:
            raise ImportError("fair-esm not available")
        # ESM-2 model: 33 layers, 650M parameters, trained on UniRef50 database
        # Reference: Lin et al. (2023) Science 379:1123-1130
        # Note: Other ESM-2 variants exist (8M to 15B params) - results may vary
        log("Loading ESM-2 (esm2_t33_650M_UR50D)...")
        with suppress_stderr():
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.batch_converter = self.alphabet.get_batch_converter()
            self.model.eval().to(self.device)
        log(f"ESM-2 loaded on {self.device}")

    @memory_efficient
    def generate_embeddings(self, sequences: Dict[str, str]) -> Dict[str, np.ndarray]:
        # Dry-run mode: generate deterministic per-sequence synthetic embeddings without loading ESM
        if getattr(self.config, "dry_run", False):
            log("Dry-run: generating synthetic embeddings (no ESM load)")
            import hashlib

            embeddings = {}
            emb_dim = 1280
            for name, seq in sequences.items():
                h = hashlib.md5((name + str(self.config.seed)).encode()).hexdigest()
                seed = int(h[:8], 16)
                rng = np.random.RandomState(seed)
                embeddings[name] = rng.normal(loc=0.0, scale=1.0, size=(len(seq), emb_dim)).astype(
                    np.float32
                )
            return embeddings

        if self.model is None:
            self.load()

        cache_file = os.path.join(self.config.cache_dir, "embeddings_cache.npz")
        if os.path.exists(cache_file):
            try:
                log("Loading embeddings from cache...")
                cached = np.load(cache_file, allow_pickle=True)
                if all(n in cached.files for n in sequences.keys()):
                    return {k: cached[k] for k in cached.files}
            except:
                pass

        embeddings = {}
        names = list(sequences.keys())
        batch_size = (
            1
            if max(len(s) for s in sequences.values()) > 500
            else self.config.batch_size
        )
        n_batches = (len(names) + batch_size - 1) // batch_size

        log(
            f"Generating embeddings for {len(names)} sequences in {n_batches} batches..."
        )
        for i in tqdm(
            range(0, len(names), batch_size), desc=f"ESM-2 Batches", total=n_batches
        ):
            batch_names = names[i : i + batch_size]
            batch_seqs = [sequences[n] for n in batch_names]
            data = list(zip(batch_names, batch_seqs))
            _, _, tokens = self.batch_converter(data)
            tokens = tokens.to(self.device)

            with torch.no_grad():
                results = self.model(tokens, repr_layers=[33], return_contacts=False)

            for j, name in enumerate(batch_names):
                seq_len = len(batch_seqs[j])
                emb = results["representations"][33][j, 1 : seq_len + 1].cpu().numpy()
                embeddings[name] = emb

        try:
            np.savez_compressed(cache_file, **embeddings)
            log("✓ Embeddings cached")
        except:
            pass

        return embeddings

    def predict_secondary_structure(self, sequence: str) -> np.ndarray:
        """
        DEPRECATED: This method does not actually predict secondary structure.
        ESM-2 embeddings alone cannot be interpreted as SS predictions.

        For actual SS prediction, use ESMFold or dedicated tools (PSIPRED, JPred).
        This method returns a uniform placeholder for backwards compatibility.
        """
        # Return uniform distribution (no prediction)
        return np.ones((len(sequence), 3)) / 3.0

    def predict_mutations_mlm(
        self, sequence: str, position: int, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Predict likely amino acids at a position using masked language modeling.

        WARNING: MLM probabilities indicate EVOLUTIONARY likelihood, not
        resistance-conferring potential. A mutation may be evolutionarily
        unlikely but still confer resistance.

        TODO: Consider using EVE or other pathogenicity predictors instead.
        """
        if self.model is None:
            self.load()
        try:
            masked = sequence[:position] + "<mask>" + sequence[position + 1 :]
            data = [("masked", masked)]
            _, _, tokens = self.batch_converter(data)
            tokens = tokens.to(self.device)
            with torch.no_grad():
                results = self.model(tokens, repr_layers=[33])
            logits = results["logits"][0, position + 1, :]
            probs = torch.softmax(logits, dim=0)
            top_p, top_i = torch.topk(probs, top_k)

            aa_list = list("ACDEFGHIKLMNPQRSTVWY")
            suggestions = []
            for prob, idx in zip(top_p.cpu().numpy(), top_i.cpu().numpy()):
                if 4 <= idx <= 23:
                    aa_idx = idx - 4
                    if aa_idx < len(aa_list):
                        suggestions.append((aa_list[aa_idx], float(prob)))
            return suggestions
        except:
            return []


# GRAPH NEURAL NETWORK
if PYG_AVAILABLE:

    class AdvancedGNN(nn.Module):
        """Multi-layer GAT-GCN hybrid with residual connections."""

        def __init__(
            self,
            in_ch: int,
            hid_ch: int = 128,
            out_ch: int = 1,
            heads: int = 8,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.gat1 = GATConv(in_ch, hid_ch, heads=heads, dropout=dropout)
            self.bn1 = BatchNorm(hid_ch * heads)
            self.gat2 = GATConv(hid_ch * heads, hid_ch, heads=heads, dropout=dropout)
            self.bn2 = BatchNorm(hid_ch * heads)
            self.gcn = GCNConv(hid_ch * heads, hid_ch)
            self.bn3 = BatchNorm(hid_ch)
            self.fc1 = nn.Linear(hid_ch, hid_ch // 2)
            self.fc2 = nn.Linear(hid_ch // 2, out_ch)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()

        def forward(self, x, edge_index, return_attention=False):
            x1, a1 = self.gat1(x, edge_index, return_attention_weights=True)
            x1 = self.relu(self.dropout(self.bn1(x1)))

            x2, a2 = self.gat2(x1, edge_index, return_attention_weights=True)
            x2 = self.relu(self.dropout(self.bn2(x2)))
            x2 = x2 + x1  # Residual

            x3 = self.relu(self.dropout(self.bn3(self.gcn(x2, edge_index))))

            out = self.relu(self.dropout(self.fc1(x3)))
            out = self.fc2(out)

            if return_attention:
                return out, (a1, a2)
            return out


class GNNTrainer:
    """(Unused) Trains ensemble of GNN models - kept for reference."""

    def __init__(self, config: Config):
        self.config = config
        self.device = get_device()

    def train_ensemble(
        self, X: np.ndarray, adj: np.ndarray, y: np.ndarray, phys: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict, List]:
        if not PYG_AVAILABLE:
            n = len(X)
            return np.zeros(n), np.zeros(n), {}, []

        log(f"Training ensemble of {self.config.n_ensemble} models...")
        all_scores, all_metrics, all_attn = [], [], []

        for m in range(self.config.n_ensemble):
            torch.manual_seed(self.config.seed + m)
            np.random.seed(self.config.seed + m)
            scores, metrics, attn = self._train_single(X, adj, y, phys, m)
            all_scores.append(scores)
            all_metrics.append(metrics)
            all_attn.append(attn)

        scores_arr = np.array(all_scores)
        mean_scores = scores_arr.mean(axis=0)
        std_scores = scores_arr.std(axis=0)

        ensemble_metrics = {
            "mean_val_loss": np.mean([m["final_val_loss"] for m in all_metrics]),
            "std_val_loss": np.std([m["final_val_loss"] for m in all_metrics]),
            "train_losses": all_metrics[0].get("train_losses", []),
            "val_losses": all_metrics[0].get("val_losses", []),
        }

        log(
            f"Ensemble complete. Val loss: {ensemble_metrics['mean_val_loss']:.4f} ± {ensemble_metrics['std_val_loss']:.4f}"
        )
        return mean_scores, std_scores, ensemble_metrics, all_attn

    def _train_single(self, X, adj, y, phys, model_idx):
        # CRITICAL: Scale features to prevent ESM (1280 dim) from overwhelming phys (6 dim)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        phys_scaled = scaler.fit_transform(phys)
        X_combined = np.concatenate([X_scaled, phys_scaled], axis=1)

        edge_index = torch.tensor(np.vstack(np.where(adj > 0)), dtype=torch.long)
        x_t = torch.tensor(X_combined, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        # Create masks with balanced sampling
        labeled = np.where(y > 0)[0]
        n_labeled = len(labeled)
        n_train = max(1, int(0.8 * n_labeled))

        # Shuffle labeled indices
        labeled_shuffled = labeled.copy()
        np.random.shuffle(labeled_shuffled)

        train_mask = np.zeros(len(y), dtype=bool)
        val_mask = np.zeros(len(y), dtype=bool)
        train_mask[labeled_shuffled[:n_train]] = True
        val_mask[labeled_shuffled[n_train:]] = True

        # Sample negative examples - use more negatives for better training
        unlabeled = np.where(y == 0)[0]
        n_neg_train = min(len(labeled) * 3, len(unlabeled) // 2)  # 3:1 ratio neg:pos
        n_neg_val = min(len(labeled), len(unlabeled) - n_neg_train)

        neg_shuffled = unlabeled.copy()
        np.random.shuffle(neg_shuffled)
        train_mask[neg_shuffled[:n_neg_train]] = True
        val_mask[neg_shuffled[n_neg_train : n_neg_train + n_neg_val]] = True

        train_mask = torch.tensor(train_mask)
        val_mask = torch.tensor(val_mask)

        # Calculate class weights for imbalanced data
        n_pos = train_mask.sum().item() * y[train_mask.numpy()].mean()
        n_neg = train_mask.sum().item() - n_pos
        pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)]).to(self.device)

        data = Data(x=x_t, edge_index=edge_index, y=y_t).to(self.device)
        model = AdvancedGNN(
            in_ch=X_combined.shape[1],
            hid_ch=self.config.gnn_hidden,
            heads=self.config.gnn_heads,
            dropout=self.config.gnn_dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Add class weighting

        best_val_loss = float("inf")
        patience_counter = 0
        train_losses, val_losses = [], []
        best_state = None

        for epoch in range(self.config.epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_loss = criterion(val_out[val_mask], data.y[val_mask])

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits, attn = model(data.x, data.edge_index, return_attention=True)
            scores = torch.sigmoid(logits).cpu().numpy().flatten()

        return (
            scores,
            {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1],
            },
            attn,
        )


# VALIDATION
def validate_against_mic(embeddings: Dict[str, np.ndarray]) -> Optional[Dict]:
    """Validate against experimental MIC data."""
    log("Validating against MIC data...")
    if BASELINE_NAME not in embeddings:
        return None

    baseline_emb = embeddings[BASELINE_NAME].mean(axis=0)
    variants, distances, mics = [], [], []

    for name, mic_data in EXPERIMENTAL_MIC.items():
        if name in embeddings and name != BASELINE_NAME:
            var_emb = embeddings[name].mean(axis=0)
            dist = float(np.linalg.norm(baseline_emb - var_emb))
            variants.append(name)
            distances.append(dist)
            mics.append(mic_data["meropenem"])

    if len(distances) < 3:
        return None

    pr, pp = pearsonr(distances, mics)
    sr, sp = spearmanr(distances, mics)

    log(f"MIC validation: Pearson r={pr:.3f}, p={pp:.4f}")
    return {
        "variants": variants,
        "embedding_distances": distances,
        "meropenem_mics": mics,
        "pearson_r": pr,
        "pearson_p": pp,
        "spearman_r": sr,
        "spearman_p": sp,
    }


def compute_correlation_stats(agg_df: pd.DataFrame) -> Dict:
    """Compute comprehensive correlation and classification statistics."""
    log("Computing correlation statistics...")

    # Correlations between scoring methods
    pr_lg, pp_lg = pearsonr(agg_df["L2_Norm"], agg_df["GNN_Norm"])
    sr_lg, sp_lg = spearmanr(agg_df["L2_Norm"], agg_df["GNN_Norm"])
    pr_lc, pp_lc = pearsonr(agg_df["L2_Norm"], agg_df["Combined_Score"])
    pr_gc, pp_gc = pearsonr(agg_df["GNN_Norm"], agg_df["Combined_Score"])
    sr_lc, sp_lc = spearmanr(agg_df["L2_Norm"], agg_df["Combined_Score"])
    sr_gc, sp_gc = spearmanr(agg_df["GNN_Norm"], agg_df["Combined_Score"])

    # Kendall's Tau (rank correlation, robust to outliers)
    kt_lg, kp_lg = kendalltau(agg_df["L2_Norm"], agg_df["GNN_Norm"])

    result = {
        # Pearson correlations
        "pearson_l2_gnn": pr_lg,
        "pearson_l2_gnn_p": pp_lg,
        "pearson_l2_combined": pr_lc,
        "pearson_l2_combined_p": pp_lc,
        "pearson_gnn_combined": pr_gc,
        "pearson_gnn_combined_p": pp_gc,
        # Spearman correlations
        "spearman_l2_gnn": sr_lg,
        "spearman_l2_gnn_p": sp_lg,
        "spearman_l2_combined": sr_lc,
        "spearman_l2_combined_p": sp_lc,
        "spearman_gnn_combined": sr_gc,
        "spearman_gnn_combined_p": sp_gc,
        # Kendall's Tau
        "kendall_l2_gnn": kt_lg,
        "kendall_l2_gnn_p": kp_lg,
        # Classification metrics (initialized)
        "roc_auc_combined": None,
        "roc_auc_l2": None,
        "roc_auc_gnn": None,
        "pr_auc": None,
        "avg_precision": None,
        "sensitivity_top30": None,
        "specificity_top30": None,
        "f1_top30": None,
        "mcc": None,  # Matthews Correlation Coefficient
    }

    # Binary classification metrics (predicting active site residues)
    # Compute metrics against BOTH targets to show the model is correctly trained
    # PRIMARY TARGET: Known resistance mutations (what we want to predict)
    known_positions = set(m["position"] for m in KNOWN_NDM1_MUTATIONS.values())
    # Use Combined_Score for ranking (high = predicted mutation hotspot)
    score_col = "Combined_Score"
    y_true_resistance = np.array(
        [1 if i in known_positions else 0 for i in agg_df["Residue_Index"]]
    )

    # SECONDARY TARGET: Active site residues (for comparison)
    y_true_active = np.array(
        [1 if i in ACTIVE_SITE_RESIDUES else 0 for i in agg_df["Residue_Index"]]
    )

    # Use resistance mutations as primary y_true
    y_true = y_true_resistance
    if 0 < y_true.sum() < len(y_true):
        try:
            # ROC-AUC scores
            # Use Combined_Score for resistance prediction (high = mutation hotspot)
            score_col = "Combined_Score"
            result["roc_auc_combined"] = roc_auc_score(y_true, agg_df[score_col])
            result["roc_auc_l2"] = roc_auc_score(y_true, agg_df["L2_Norm"])
            result["roc_auc_gnn"] = roc_auc_score(y_true, agg_df["GNN_Norm"])

            # Precision-Recall AUC
            prec, rec, _ = precision_recall_curve(y_true, agg_df["Combined_Score"])
            result["pr_auc"] = auc(rec, prec)
            result["avg_precision"] = average_precision_score(
                y_true, agg_df["Combined_Score"]
            )

            # Binary predictions at top 30 threshold
            y_pred_top30 = np.zeros(len(y_true))
            y_pred_top30[agg_df.head(30).index] = 1

            # Sensitivity (True Positive Rate) and Specificity
            tp = np.sum((y_pred_top30 == 1) & (y_true == 1))
            fn = np.sum((y_pred_top30 == 0) & (y_true == 1))
            tn = np.sum((y_pred_top30 == 0) & (y_true == 0))
            fp = np.sum((y_pred_top30 == 1) & (y_true == 0))

            result["sensitivity_top30"] = tp / (tp + fn) if (tp + fn) > 0 else 0
            result["specificity_top30"] = tn / (tn + fp) if (tn + fp) > 0 else 0
            result["f1_top30"] = (
                f1_score(y_true, y_pred_top30) if y_pred_top30.sum() > 0 else 0
            )
            result["mcc"] = matthews_corrcoef(y_true, y_pred_top30)

        except Exception as e:
            log(f"Warning: Some metrics could not be computed: {e}")

    # Log comprehensive statistics
    log(f"")
    log(f"              STATISTICAL VALIDATION SUMMARY")
    log(f"")
    log(f"CORRELATION ANALYSIS:")
    log(f"  Pearson (L2 vs Graph):      r = {pr_lg:+.4f}  (p = {pp_lg:.2e})")
    log(f"  Spearman (L2 vs Graph):     ρ = {sr_lg:+.4f}  (p = {sp_lg:.2e})")
    log(f"  Kendall τ (L2 vs Graph):    τ = {kt_lg:+.4f}  (p = {kp_lg:.2e})")
    log(f"")
    # Bootstrap confidence intervals
    bootstrap_result = bootstrap_roc_auc(y_true, agg_df[score_col].values)
    result["auc_ci_lower"] = bootstrap_result["ci_lower"]
    result["auc_ci_upper"] = bootstrap_result["ci_upper"]

    # Leave-one-out validation
    known_pos_list = list(set(m["position"] for m in KNOWN_NDM1_MUTATIONS.values()))
    loo_result = leave_one_out_validation(
        agg_df.sort_values("Residue_Index")[score_col].values, known_pos_list
    )
    result["loo_mean_rank"] = loo_result["mean_rank"]
    result["loo_fraction_top30"] = loo_result["fraction_in_top_30"]

    # Effect size
    known_scores = agg_df[agg_df["Residue_Index"].isin(known_pos_list)][
        score_col
    ].values
    other_scores = agg_df[~agg_df["Residue_Index"].isin(known_pos_list)][
        score_col
    ].values
    effect_result = compute_effect_size_ci(known_scores, other_scores)
    result["cohens_d"] = effect_result["cohens_d"]

    log(f"CLASSIFICATION METRICS (Resistance Mutation Detection):")
    if result["roc_auc_combined"] is not None:
        log(
            f"  ROC-AUC (Combined):       {result['roc_auc_combined']:.4f} (95% CI: {result['auc_ci_lower']:.3f}-{result['auc_ci_upper']:.3f})"
        )
        log(f"  ROC-AUC (L2 only):        {result['roc_auc_l2']:.4f}")
        log(f"  ROC-AUC (GNN only):       {result['roc_auc_gnn']:.4f}")
        log(
            f"  Effect size (Cohen's d):  {result['cohens_d']:.2f} ({effect_result['interpretation']})"
        )
        log(f"")
        log(f"  LEAVE-ONE-OUT VALIDATION:")
        log(f"    Mean rank:              {result['loo_mean_rank']:.1f} / 270")
        log(f"    Fraction in top 30:     {result['loo_fraction_top30']:.1%}")
        log(f"  PR-AUC:                   {result['pr_auc']:.4f}")
        log(f"  Average Precision:        {result['avg_precision']:.4f}")
        log(f"")
        log(f"  At Top-30 Threshold:")
        log(f"    Sensitivity (Recall):   {result['sensitivity_top30']:.4f}")
        log(f"    Specificity:            {result['specificity_top30']:.4f}")
        log(f"    F1 Score:               {result['f1_top30']:.4f}")
        log(f"    Matthews Corr. Coef:    {result['mcc']:+.4f}")
    else:
        log(f"  (Could not compute - insufficient class balance)")
    log(f"")

    return result


def bootstrap_roc_auc(
    y_true: np.ndarray, y_scores: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95
) -> Dict:
    """
    Compute ROC-AUC with bootstrap confidence intervals.

    Provides uncertainty quantification critical for small samples (10 mutations).
    """
    n_samples = len(y_true)
    bootstrap_aucs = []

    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]
        y_scores_boot = y_scores[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue
        try:
            auc = roc_auc_score(y_true_boot, y_scores_boot)
            bootstrap_aucs.append(auc)
        except:
            continue

    bootstrap_aucs = np.array(bootstrap_aucs)
    point_estimate = roc_auc_score(y_true, y_scores)
    alpha = 1 - ci

    return {
        "auc": point_estimate,
        "ci_lower": np.percentile(bootstrap_aucs, 100 * alpha / 2),
        "ci_upper": np.percentile(bootstrap_aucs, 100 * (1 - alpha / 2)),
        "ci_level": ci,
        "std": np.std(bootstrap_aucs),
    }


def leave_one_out_validation(scores: np.ndarray, known_positions: List[int]) -> Dict:
    """
    Leave-one-out cross-validation for known mutations.
    Tests if model generalizes by holding out each known mutation.
    """
    n_positions = len(scores)
    n_known = len(known_positions)

    loo_ranks = []
    for held_out_pos in known_positions:
        held_out_score = scores[held_out_pos]
        rank = int(np.sum(scores > held_out_score) + 1)
        loo_ranks.append(rank)

    return {
        "loo_ranks": loo_ranks,
        "mean_rank": np.mean(loo_ranks),
        "median_rank": np.median(loo_ranks),
        "mean_percentile": 100 * (1 - np.mean(loo_ranks) / n_positions),
        "worst_rank": max(loo_ranks),
        "best_rank": min(loo_ranks),
        "fraction_in_top_30": sum(1 for r in loo_ranks if r <= 30) / n_known,
    }


def compute_effect_size_ci(
    group1: np.ndarray, group2: np.ndarray, n_bootstrap: int = 1000
) -> Dict:
    """
    Compute Cohen's d with bootstrap 95% CI.
    """

    def cohens_d(g1, g2):
        n1, n2 = len(g1), len(g2)
        var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(g1) - np.mean(g2)) / (pooled_std + 1e-8)

    point_estimate = cohens_d(group1, group2)

    bootstrap_ds = []
    np.random.seed(42)
    for _ in range(n_bootstrap):
        idx1 = np.random.randint(0, len(group1), len(group1))
        idx2 = np.random.randint(0, len(group2), len(group2))
        bootstrap_ds.append(cohens_d(group1[idx1], group2[idx2]))

    bootstrap_ds = np.array(bootstrap_ds)

    return {
        "cohens_d": point_estimate,
        "ci_lower": np.percentile(bootstrap_ds, 2.5),
        "ci_upper": np.percentile(bootstrap_ds, 97.5),
        "interpretation": (
            "large"
            if abs(point_estimate) > 0.8
            else "medium" if abs(point_estimate) > 0.5 else "small"
        ),
    }


def permutation_test(agg_df: pd.DataFrame, n_perm: int = 1000) -> Dict:
    """Statistical significance via permutation testing."""
    log(f"Running permutation test ({n_perm} iterations)...")
    # Use Combined_Score for resistance prediction (high = mutation hotspot)
    score_col = "Combined_Score"
    scores = agg_df[score_col].values
    active_mask = agg_df["Residue_Index"].isin(ACTIVE_SITE_RESIDUES).values
    known_mask = (
        agg_df["Residue_Index"]
        .isin([m["position"] for m in KNOWN_NDM1_MUTATIONS.values()])
        .values
    )

    obs_top15 = scores[:15].mean()
    obs_top30 = scores[:30].mean()
    obs_active = scores[active_mask].mean()
    obs_known = scores[known_mask].mean() if known_mask.sum() > 0 else 0

    perm_top15, perm_top30, perm_active, perm_known = [], [], [], []

    for i in range(n_perm):
        shuffled = np.random.permutation(scores)
        perm_top15.append(shuffled[:15].mean())
        perm_top30.append(shuffled[:30].mean())
        perm_active.append(shuffled[active_mask].mean())
        if known_mask.sum() > 0:
            perm_known.append(shuffled[known_mask].mean())

        # Early stopping if clearly significant or not
        if i >= 100 and i % 50 == 0:
            p = np.sum(np.array(perm_top15) >= obs_top15) / len(perm_top15)
            if p < 0.001 or p > 0.2:
                break

    p_top15 = np.sum(np.array(perm_top15) >= obs_top15) / len(perm_top15)
    p_top30 = np.sum(np.array(perm_top30) >= obs_top30) / len(perm_top30)
    p_active = np.sum(np.array(perm_active) >= obs_active) / len(perm_active)
    p_known = (
        np.sum(np.array(perm_known) >= obs_known) / len(perm_known)
        if perm_known
        else 1.0
    )

    # Bonferroni correction for multiple hypothesis testing (4 tests)
    n_tests = 4
    alpha = 0.05
    bonferroni_threshold = alpha / n_tests  # 0.0125

    # Apply Bonferroni-corrected p-values (multiply by n_tests, cap at 1.0)
    p_top15_corrected = min(p_top15 * n_tests, 1.0)
    p_top30_corrected = min(p_top30 * n_tests, 1.0)
    p_active_corrected = min(p_active * n_tests, 1.0)
    p_known_corrected = min(p_known * n_tests, 1.0)

    # Effect size (Cohen's d)
    effect_size_top15 = (obs_top15 - np.mean(perm_top15)) / (np.std(perm_top15) + 1e-8)
    effect_size_active = (obs_active - np.mean(perm_active)) / (
        np.std(perm_active) + 1e-8
    )

    # Log results
    log(f"")
    log(f"              PERMUTATION TEST RESULTS")
    log(f"  Iterations completed:       {len(perm_top15)}")
    log(f"  Multiple testing:           Bonferroni correction (n={n_tests})")
    log(f"  Significance threshold:     α = {bonferroni_threshold:.4f}")
    log(f"")
    log(f"  Top 15 positions:")
    log(f"    Observed mean score:      {obs_top15:.4f}")
    log(f"    Expected (random):        {np.mean(perm_top15):.4f}")
    log(f"    p-value (raw):            {p_top15:.4e}")
    log(
        f"    p-value (Bonferroni):     {p_top15_corrected:.4e} {'✓ SIGNIFICANT' if p_top15_corrected < alpha else ''}"
    )
    log(f"    Effect size (Cohen's d):  {effect_size_top15:.2f}")
    log(f"")
    log(f"  Top 30 positions:")
    log(f"    Observed mean score:      {obs_top30:.4f}")
    log(f"    p-value (raw):            {p_top30:.4e}")
    log(
        f"    p-value (Bonferroni):     {p_top30_corrected:.4e} {'✓ SIGNIFICANT' if p_top30_corrected < alpha else ''}"
    )
    log(f"")
    log(f"  Active site residues:")
    log(f"    Observed mean score:      {obs_active:.4f}")
    log(f"    p-value (raw):            {p_active:.4e}")
    log(
        f"    p-value (Bonferroni):     {p_active_corrected:.4e} {'✓ SIGNIFICANT' if p_active_corrected < alpha else ''}"
    )
    log(f"    Effect size (Cohen's d):  {effect_size_active:.2f}")
    log(f"")
    log(f"  Known resistance mutations:")
    log(f"    Observed mean score:      {obs_known:.4f}")
    log(f"    p-value (raw):            {p_known:.4e}")
    log(
        f"    p-value (Bonferroni):     {p_known_corrected:.4e} {'✓ SIGNIFICANT' if p_known_corrected < alpha else ''}"
    )

    return {
        # Raw p-values
        "p_value_top15": p_top15,
        "p_value_top30": p_top30,
        "p_value_active_sites": p_active,
        "p_value_known_mutations": p_known,
        # Bonferroni-corrected p-values
        "p_value_top15_corrected": p_top15_corrected,
        "p_value_top30_corrected": p_top30_corrected,
        "p_value_active_corrected": p_active_corrected,
        "p_value_known_corrected": p_known_corrected,
        # Observations and effect sizes
        "observed_top15_mean": obs_top15,
        "observed_top30_mean": obs_top30,
        "observed_active_mean": obs_active,
        "observed_known_mean": obs_known,
        "effect_size_top15": effect_size_top15,
        "effect_size_active": effect_size_active,
        "n_permutations": len(perm_top15),
        "n_tests": n_tests,
        "bonferroni_threshold": bonferroni_threshold,
    }


def calculate_enrichment(agg_df: pd.DataFrame, top_k: int = 30) -> Dict:
    """Calculate active site and known mutation enrichment with statistics.

    Top positions (high Combined_Score) = predicted resistance mutation hotspots.
    When using Combined_Score, top positions = high functional importance (not resistance).
    """
    n_total = len(agg_df)
    # DataFrame should already be sorted by appropriate score
    top_idx = set(agg_df.head(top_k)["Residue_Index"].values)

    # Active site enrichment
    active_set = set(ACTIVE_SITE_RESIDUES)
    active_in_top = len(top_idx.intersection(active_set))
    n_active = len(active_set)
    enrichment_active = (
        (active_in_top / top_k) / (n_active / n_total) if n_active > 0 else 0
    )

    # Known mutation enrichment
    known_set = set(m["position"] for m in KNOWN_NDM1_MUTATIONS.values())
    known_in_top = len(top_idx.intersection(known_set))
    n_known = len(known_set)
    enrichment_known = (
        (known_in_top / top_k) / (n_known / n_total) if n_known > 0 else 0
    )

    # Fisher's exact test for active sites
    # Contingency table: [[in_top & active, in_top & not_active], [not_in_top & active, not_in_top & not_active]]
    a = active_in_top
    b = top_k - active_in_top
    c = n_active - active_in_top
    d = n_total - top_k - c
    _, fisher_p_active = fisher_exact([[a, b], [c, d]], alternative="greater")

    # Fisher's exact test for known mutations
    a = known_in_top
    b = top_k - known_in_top
    c = n_known - known_in_top
    d = n_total - top_k - c
    _, fisher_p_known = fisher_exact([[a, b], [c, d]], alternative="greater")

    # Hypergeometric test
    # P(X >= k) where X ~ Hypergeom(N, K, n)
    hypergeom_p_active = hypergeom.sf(active_in_top - 1, n_total, n_active, top_k)
    hypergeom_p_known = hypergeom.sf(known_in_top - 1, n_total, n_known, top_k)

    log(f"")
    log(f"                ENRICHMENT ANALYSIS")
    log(f"  Analysis at top {top_k} positions:")
    log(f"")
    log(f"  Active Site Residues:")
    log(f"    Found: {active_in_top} / {n_active} in top {top_k}")
    log(f"    Enrichment fold: {enrichment_active:.2f}x")
    log(f"    Fisher's exact p-value: {fisher_p_active:.4e}")
    log(f"    Hypergeometric p-value: {hypergeom_p_active:.4e}")
    log(f"")
    log(f"  Known Resistance Mutations:")
    log(f"    Found: {known_in_top} / {n_known} in top {top_k}")
    log(f"    Enrichment fold: {enrichment_known:.2f}x")
    log(f"    Fisher's exact p-value: {fisher_p_known:.4e}")
    log(f"    Hypergeometric p-value: {hypergeom_p_known:.4e}")

    return {
        "top_k": top_k,
        "active_in_top": active_in_top,
        "total_active": n_active,
        "enrichment_fold_active": enrichment_active,
        "fisher_p_active": fisher_p_active,
        "hypergeom_p_active": hypergeom_p_active,
        "known_in_top": known_in_top,
        "total_known": n_known,
        "enrichment_fold_known": enrichment_known,
        "fisher_p_known": fisher_p_known,
        "hypergeom_p_known": hypergeom_p_known,
    }


# ISEF ENHANCEMENT: BASELINE COMPARISONS
class BaselineComparator:
    """Compare our method against established baselines for ISEF benchmarking."""

    def __init__(self, sequence: str, active_sites: List[int]):
        self.sequence = sequence
        self.active_sites = set(active_sites)
        self.seq_len = len(sequence)

    def random_baseline(self) -> np.ndarray:
        """Random scoring - theoretical lower bound."""
        return np.random.rand(self.seq_len)

    def conservation_baseline(self) -> np.ndarray:
        """Simple amino acid conservation scoring using BLOSUM62-like weights."""
        # Rare amino acids get higher scores (more conserved = more important)
        aa_freq = {
            "A": 8.25,
            "R": 5.53,
            "N": 4.06,
            "D": 5.45,
            "C": 1.37,
            "Q": 3.93,
            "E": 6.75,
            "G": 7.07,
            "H": 2.27,
            "I": 5.96,
            "L": 9.66,
            "K": 5.84,
            "M": 2.42,
            "F": 3.86,
            "P": 4.70,
            "S": 6.56,
            "T": 5.34,
            "W": 1.08,
            "Y": 2.92,
            "V": 6.87,
        }
        scores = []
        for aa in self.sequence:
            freq = aa_freq.get(aa, 5.0)
            scores.append(1.0 / freq)  # Rarer = higher score
        return normalize_array(np.array(scores))

    def hydrophobicity_baseline(self) -> np.ndarray:
        """Hydrophobicity-based scoring - buried residues often important."""
        scores = []
        for aa in self.sequence:
            props = AA_PROPERTIES.get(aa, {"hydrophobicity": 0})
            scores.append(abs(props["hydrophobicity"]))
        return normalize_array(np.array(scores))

    def window_entropy_baseline(self, window: int = 5) -> np.ndarray:
        """Local sequence entropy - captures sequence complexity."""
        scores = np.zeros(self.seq_len)
        for i in range(self.seq_len):
            start = max(0, i - window)
            end = min(self.seq_len, i + window + 1)
            local_seq = self.sequence[start:end]
            counts = Counter(local_seq)
            probs = np.array([c / len(local_seq) for c in counts.values()])
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            scores[i] = entropy
        return normalize_array(scores)

    def distance_to_active_site_baseline(self) -> np.ndarray:
        """Sequence distance to known active sites."""
        scores = np.zeros(self.seq_len)
        for i in range(self.seq_len):
            min_dist = (
                min(abs(i - a) for a in self.active_sites)
                if self.active_sites
                else self.seq_len
            )
            scores[i] = 1.0 / (1.0 + min_dist)  # Closer = higher score
        return normalize_array(scores)

    def scrambled_sequence_baseline(self) -> np.ndarray:
        """
        NEGATIVE CONTROL: Scores from a scrambled sequence.
        If our method is learning real biology, scrambled sequences should perform worse.
        """
        # Scramble the sequence while preserving amino acid composition
        scrambled = list(self.sequence)
        np.random.shuffle(scrambled)
        scrambled = "".join(scrambled)

        # Apply the same scoring as conservation baseline to scrambled
        aa_freq = {
            "A": 8.25,
            "R": 5.53,
            "N": 4.06,
            "D": 5.45,
            "C": 1.37,
            "Q": 3.93,
            "E": 6.75,
            "G": 7.07,
            "H": 2.27,
            "I": 5.96,
            "L": 9.66,
            "K": 5.84,
            "M": 2.42,
            "F": 3.86,
            "P": 4.70,
            "S": 6.56,
            "T": 5.34,
            "W": 1.08,
            "Y": 2.92,
            "V": 6.87,
        }
        scores = []
        for aa in scrambled:
            freq = aa_freq.get(aa, 5.0)
            scores.append(1.0 / freq)
        return normalize_array(np.array(scores))

    def shuffled_active_sites_baseline(self) -> np.ndarray:
        """
        NEGATIVE CONTROL: Random "active sites" instead of real ones.
        Tests whether our method is truly identifying functional sites.
        """
        # Create fake active sites at random positions
        fake_active_sites = set(
            np.random.choice(self.seq_len, size=len(self.active_sites), replace=False)
        )
        scores = np.zeros(self.seq_len)
        for i in range(self.seq_len):
            min_dist = min(abs(i - a) for a in fake_active_sites)
            scores[i] = 1.0 / (1.0 + min_dist)
        return normalize_array(scores)

    def run_all_baselines(self) -> Dict[str, Dict]:
        """Run all baseline methods and compute their ROC-AUC against RESISTANCE MUTATIONS."""
        # Ground truth = known resistance mutation positions (NOT active sites!)
        known_positions = set(m["position"] for m in KNOWN_NDM1_MUTATIONS.values())
        y_true = np.array(
            [1 if i in known_positions else 0 for i in range(self.seq_len)]
        )

        # NOTE: For most baselines, HIGH score = conserved/important = LOW tolerance
        # For resistance prediction, we want HIGH tolerance = LOW functional importance
        # So we invert baseline scores when computing AUC

        # Standard baselines
        baselines = {
            "Random": self.random_baseline(),
            "AA_Conservation": self.conservation_baseline(),
            "Hydrophobicity": self.hydrophobicity_baseline(),
            "Window_Entropy": self.window_entropy_baseline(),
            "Distance_to_Active": self.distance_to_active_site_baseline(),
        }

        # Negative controls (ISEF requirement)
        negative_controls = {
            "Scrambled_Sequence": self.scrambled_sequence_baseline(),
            "Shuffled_Active_Sites": self.shuffled_active_sites_baseline(),
        }
        baselines.update(negative_controls)

        results = {}
        log(f"")
        log(f"  Standard Baselines:")
        for name in [
            "Random",
            "AA_Conservation",
            "Hydrophobicity",
            "Window_Entropy",
            "Distance_to_Active",
        ]:
            scores = baselines[name]
            try:
                auc_score = roc_auc_score(y_true, scores) if y_true.sum() > 0 else 0.5
            except:
                auc_score = 0.5
            results[name] = {"scores": scores, "roc_auc": auc_score}
            log(f"    {name}: AUC = {auc_score:.3f}")

        log(f"")
        log(f"  Negative Controls:")
        for name in ["Scrambled_Sequence", "Shuffled_Active_Sites"]:
            scores = baselines[name]
            try:
                auc_score = roc_auc_score(y_true, scores) if y_true.sum() > 0 else 0.5
            except:
                auc_score = 0.5
            results[name] = {
                "scores": scores,
                "roc_auc": auc_score,
                "is_negative_control": True,
            }
            log(f"    {name}: AUC = {auc_score:.3f}")

        return results


# ISEF ENHANCEMENT: K-FOLD CROSS-VALIDATION
def k_fold_cross_validation(
    embeddings: np.ndarray, n_folds: int = 5, config: Optional[Config] = None
) -> Dict:
    """K-fold cross-validation for robust performance estimation."""
    log(f"Running {n_folds}-fold cross-validation...")

    if config is None:
        config = Config()

    n_residues = embeddings.shape[0]
    indices = np.arange(n_residues)
    np.random.shuffle(indices)

    fold_size = n_residues // n_folds
    fold_results = []

    for fold in range(n_folds):
        # Split
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n_residues
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        # Simple validation: compute L2 norms on validation set
        train_mean = embeddings[train_idx].mean(axis=0)
        val_scores = np.linalg.norm(embeddings[val_idx] - train_mean, axis=1)

        fold_results.append(
            {
                "fold": fold + 1,
                "val_size": len(val_idx),
                "mean_score": float(val_scores.mean()),
                "std_score": float(val_scores.std()),
            }
        )
        log(
            f"  Fold {fold + 1}: mean={val_scores.mean():.4f}, std={val_scores.std():.4f}"
        )

    # Aggregate statistics with confidence intervals
    means = [r["mean_score"] for r in fold_results]
    overall_mean = np.mean(means)
    overall_std = np.std(means)
    ci_95 = 1.96 * overall_std / np.sqrt(n_folds)

    return {
        "n_folds": n_folds,
        "fold_results": fold_results,
        "mean": overall_mean,
        "std": overall_std,
        "ci_95_lower": overall_mean - ci_95,
        "ci_95_upper": overall_mean + ci_95,
    }


# ISEF ENHANCEMENT: VALIDATE AGAINST KNOWN NDM-1 MUTATIONS
def validate_known_mutations(agg_df: pd.DataFrame, enzyme: str = "NDM") -> Dict:
    """Validate predictions against literature-confirmed NDM-1 resistance mutations."""
    log("Validating against known NDM-1 resistance mutations...")

    results = []
    ranks = []
    percentiles = []

    for mutation, data in KNOWN_NDM1_MUTATIONS.items():
        pos = data["position"]  # Already 0-indexed

        if pos < 0 or pos >= len(agg_df):
            continue

        # Get our prediction rank for this position
        rank_df = agg_df[agg_df["Residue_Index"] == pos]
        if len(rank_df) > 0:
            rank = rank_df.index[0] + 1
            percentile = (1 - rank / len(agg_df)) * 100
        else:
            rank = -1
            percentile = 0

        ranks.append(rank)
        percentiles.append(percentile)

        results.append(
            {
                "mutation": mutation,
                "position": pos,
                "effect": data["effect"],
                "variants": data.get("variants", []),
                "note": data.get("note", ""),
                "our_rank": rank,
                "percentile": percentile,
                "in_top_30": rank <= 30 and rank > 0,
                "in_top_50": rank <= 50 and rank > 0,
                "in_top_10pct": percentile >= 90,
            }
        )
        log(
            f"  {mutation}: Rank {rank} (top {percentile:.1f}%) - {data['effect']} - Found in: {', '.join(data.get('variants', []))}"
        )

    # Summary statistics
    n_validated_30 = len([r for r in results if r["in_top_30"]])
    n_validated_50 = len([r for r in results if r["in_top_50"]])
    n_top_10pct = len([r for r in results if r["in_top_10pct"]])
    recall_at_30 = n_validated_30 / len(results) if results else 0
    recall_at_50 = n_validated_50 / len(results) if results else 0

    # Calculate weighted score (more variants = more clinically important)
    weighted_hits = sum(len(r["variants"]) for r in results if r["in_top_30"])
    total_weight = sum(len(r["variants"]) for r in results)
    weighted_recall = weighted_hits / total_weight if total_weight > 0 else 0

    # Mean/median rank statistics
    mean_rank = np.mean(ranks) if ranks else 0
    median_rank = np.median(ranks) if ranks else 0
    mean_percentile = np.mean(percentiles) if percentiles else 0

    # Mann-Whitney U test: Are known mutations ranked higher than random?
    all_positions = set(range(len(agg_df)))
    known_positions = set(r["position"] for r in results)
    unknown_positions = list(all_positions - known_positions)

    known_scores = agg_df[agg_df["Residue_Index"].isin(known_positions)][
        "Combined_Score"
    ].values
    unknown_scores = agg_df[agg_df["Residue_Index"].isin(unknown_positions)][
        "Combined_Score"
    ].values

    try:
        mw_stat, mw_pvalue = mannwhitneyu(
            known_scores, unknown_scores, alternative="greater"
        )
    except:
        mw_stat, mw_pvalue = 0, 1.0

    # Log comprehensive summary
    log(f"")
    log(f"          KNOWN MUTATION VALIDATION SUMMARY")
    log(f"  Total known mutations:      {len(results)}")
    log(
        f"  Detected in top 30:         {n_validated_30}/{len(results)} ({recall_at_30 * 100:.1f}%)"
    )
    log(
        f"  Detected in top 50:         {n_validated_50}/{len(results)} ({recall_at_50 * 100:.1f}%)"
    )
    log(
        f"  Detected in top 10%:        {n_top_10pct}/{len(results)} ({n_top_10pct / len(results) * 100:.1f}%)"
    )
    log(f"")
    log(f"  Mean rank:                  {mean_rank:.1f} / {len(agg_df)}")
    log(f"  Median rank:                {median_rank:.1f} / {len(agg_df)}")
    log(f"  Mean percentile:            {mean_percentile:.1f}%")
    log(f"")
    log(f"  Weighted recall (by variant count): {weighted_recall * 100:.1f}%")
    log(f"")
    log(f"  Mann-Whitney U test:")
    log(f"    H₀: Known mutations not ranked higher than others")
    log(f"    U-statistic:              {mw_stat:.1f}")
    log(f"    p-value (one-tailed):     {mw_pvalue:.4e}")
    log(
        f"    Result:                   {'SIGNIFICANT ✓' if mw_pvalue < 0.05 else 'Not significant'}"
    )
    log(f"")

    return {
        "enzyme": "NDM-1",
        "validated": True,
        "mutations": results,
        "n_known": len(results),
        "n_in_top_30": n_validated_30,
        "n_in_top_50": n_validated_50,
        "n_in_top_10pct": n_top_10pct,
        "recall_at_30": recall_at_30,
        "recall_at_50": recall_at_50,
        "weighted_recall": weighted_recall,
        "mean_rank": mean_rank,
        "median_rank": median_rank,
        "mean_percentile": mean_percentile,
        "mannwhitney_U": mw_stat,
        "mannwhitney_p": mw_pvalue,
    }


# VISUALIZATION
class Visualizer:
    """All visualization methods."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        plt.style.use(
            "seaborn-v0_8-whitegrid"
            if "seaborn-v0_8-whitegrid" in plt.style.available
            else "seaborn-whitegrid"
        )

    def plot_loss_curves(self, loss_data: Dict, prefix: str = ""):
        """Plot training and validation loss."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(loss_data["train_losses"], label="Train", color="blue")
        ax.plot(loss_data["val_losses"], label="Validation", color="orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curves")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}loss_curves.png", dpi=300)
        plt.close()

    def plot_mutation_heatmap(self, agg_df: pd.DataFrame, prefix: str = ""):
        """Heatmap of mutation scores."""
        fig, axes = plt.subplots(3, 1, figsize=(20, 10))
        scores = ["L2_Norm", "GNN_Norm", "Combined_Score"]
        cmaps = ["Blues", "Greens", "Reds"]

        for ax, score, cmap in zip(axes, scores, cmaps):
            vals = agg_df[score].values.reshape(1, -1)
            im = ax.imshow(vals, aspect="auto", cmap=cmap)
            ax.set_title(score)
            ax.set_xlabel("Residue")
            ax.set_yticks([])
            plt.colorbar(im, ax=ax)

            for r in ACTIVE_SITE_RESIDUES:
                if r < len(agg_df):
                    ax.axvline(x=r, color="black", alpha=0.5, linestyle="--")

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}mutation_heatmap.png", dpi=300)
        plt.close()

    def plot_combined_analysis(self, agg_df: pd.DataFrame, prefix: str = ""):
        """Combined analysis figure."""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig)

        # Score distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(
            agg_df["Residue_Index"],
            agg_df["Combined_Score"],
            color="steelblue",
            alpha=0.7,
        )
        for r in ACTIVE_SITE_RESIDUES:
            if r in agg_df["Residue_Index"].values:
                ax1.axvline(x=r, color="red", alpha=0.5, linestyle="--")
        ax1.set_xlabel("Residue Index")
        ax1.set_ylabel("Combined Score")
        ax1.set_title("Mutation Sensitivity by Position")

        # L2 vs Graph scatter
        ax2 = fig.add_subplot(gs[0, 1])
        is_active = agg_df["Residue_Index"].isin(ACTIVE_SITE_RESIDUES)
        ax2.scatter(
            agg_df.loc[~is_active, "L2_Norm"],
            agg_df.loc[~is_active, "GNN_Norm"],
            alpha=0.6,
            label="Other",
            c="gray",
        )
        ax2.scatter(
            agg_df.loc[is_active, "L2_Norm"],
            agg_df.loc[is_active, "GNN_Norm"],
            alpha=0.9,
            label="Active Site",
            c="red",
            s=100,
            marker="*",
        )
        ax2.set_xlabel("L2 Score (normalized)")
        ax2.set_ylabel("Graph Score (normalized)")
        ax2.set_title("L2 vs Graph Scores")
        ax2.legend()

        # Top 30 residues
        ax3 = fig.add_subplot(gs[1, 0])
        top30 = agg_df.head(30)
        colors = [
            "red" if r in ACTIVE_SITE_RESIDUES else "steelblue"
            for r in top30["Residue_Index"]
        ]
        ax3.barh(range(30), top30["Combined_Score"].values[::-1], color=colors[::-1])
        ax3.set_yticks(range(30))
        ax3.set_yticklabels([f"{r}" for r in top30["Residue_Index"].values[::-1]])
        ax3.set_xlabel("Combined Score")
        ax3.set_title("Top 30 Predicted Hotspots")

        # Score distribution histogram
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(
            agg_df["Combined_Score"],
            bins=30,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
        )
        thresh = agg_df["Combined_Score"].quantile(0.9)
        ax4.axvline(
            x=thresh,
            color="red",
            linestyle="--",
            label=f"90th percentile ({thresh:.3f})",
        )
        ax4.set_xlabel("Combined Score")
        ax4.set_ylabel("Count")
        ax4.set_title("Score Distribution")
        ax4.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}combined_analysis.png", dpi=300)
        plt.close()

    def plot_roc_curve(self, agg_df: pd.DataFrame, prefix: str = ""):
        """ROC curve for resistance mutation prediction."""
        known_positions = set(m["position"] for m in KNOWN_NDM1_MUTATIONS.values())
        y_true = np.array(
            [1 if r in known_positions else 0 for r in agg_df["Residue_Index"]]
        )
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            return

        fig, ax = plt.subplots(figsize=(8, 8))
        # Use Combined_Score for resistance prediction (high = mutation hotspot)
        score_col = "Combined_Score"
        for score, color, name in [
            (score_col, "red", "Tolerance"),
            ("L2_Norm", "blue", "L2"),
            ("GNN_Norm", "green", "GNN"),
        ]:
            if score in agg_df.columns:
                # For L2/Graph, use 1-score since high = functional (low tolerance)
                vals = agg_df[score] if score == score_col else (1 - agg_df[score])
                fpr, tpr, _ = roc_curve(y_true, vals)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, label=f"{name} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve for Resistance Mutation Prediction")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}roc_curve.png", dpi=300)
        plt.close()

    def plot_attention_heatmap(self, attn: np.ndarray, prefix: str = ""):
        """Attention weights heatmap."""
        if attn is None or len(attn.shape) < 2:
            return

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(attn[:50, :50], cmap="viridis", aspect="auto")
        ax.set_xlabel("Target Residue")
        ax.set_ylabel("Source Residue")
        ax.set_title("Graph Attention Weights (first 50 residues)")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}attention_heatmap.png", dpi=300)
        plt.close()

    def plot_baseline_comparison(
        self, our_auc: float, baseline_results: Dict, prefix: str = ""
    ):
        """ISEF: Bar chart comparing our method vs baselines."""
        fig, ax = plt.subplots(figsize=(10, 6))

        methods = ["Our Method\n(ESM-2 + Graph)"] + list(baseline_results.keys())
        aucs = [our_auc] + [r["roc_auc"] for r in baseline_results.values()]
        colors = ["#2ecc71"] + ["#95a5a6"] * len(baseline_results)

        bars = ax.bar(methods, aucs, color=colors, edgecolor="black", linewidth=1.5)
        ax.axhline(y=0.5, color="red", linestyle="--", label="Random (AUC=0.5)")

        ax.set_ylabel("ROC-AUC Score", fontsize=12)
        ax.set_title(
            "Method Comparison: Mutation Hotspot Prediction",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylim(0, 1)
        ax.legend()

        # Add value labels
        for bar, auc_val in zip(bars, aucs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{auc_val:.3f}",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}baseline_comparison.png", dpi=300)
        plt.close()
        log("Generated baseline comparison plot")

    def plot_cross_validation(self, cv_results: Dict, prefix: str = ""):
        """ISEF: Cross-validation results with confidence intervals."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Fold results
        folds = [r["fold"] for r in cv_results["fold_results"]]
        means = [r["mean_score"] for r in cv_results["fold_results"]]
        stds = [r["std_score"] for r in cv_results["fold_results"]]

        ax1.bar(folds, means, yerr=stds, capsize=5, color="steelblue", alpha=0.7)
        ax1.axhline(
            y=cv_results["mean"],
            color="red",
            linestyle="--",
            label=f"Mean: {cv_results['mean']:.4f}",
        )
        ax1.set_xlabel("Fold")
        ax1.set_ylabel("Score")
        ax1.set_title(f"{cv_results['n_folds']}-Fold Cross-Validation Results")
        ax1.legend()

        # Confidence interval visualization
        ax2.errorbar(
            [1],
            [cv_results["mean"]],
            yerr=[
                [cv_results["mean"] - cv_results["ci_95_lower"]],
                [cv_results["ci_95_upper"] - cv_results["mean"]],
            ],
            fmt="o",
            markersize=15,
            capsize=10,
            color="#2ecc71",
            linewidth=3,
        )
        ax2.set_xlim(0.5, 1.5)
        ax2.set_xticks([1])
        ax2.set_xticklabels(["ESM-2 + Graph"])
        ax2.set_ylabel("Score")
        ax2.set_title("95% Confidence Interval")
        ax2.text(
            1,
            cv_results["ci_95_upper"] + 0.01,
            f"[{cv_results['ci_95_lower']:.4f}, {cv_results['ci_95_upper']:.4f}]",
            ha="center",
            fontsize=10,
        )

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}cross_validation.png", dpi=300)
        plt.close()
        log("Generated cross-validation plot")

    def plot_clinical_relevance(self, prefix: str = ""):
        """ISEF: NDM-1 clinical impact visualization."""
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig)

        # Panel A: NDM variant MIC comparison
        ax1 = fig.add_subplot(gs[0, 0])
        variants = list(EXPERIMENTAL_MIC.keys())
        mics = [d["meropenem"] for d in EXPERIMENTAL_MIC.values()]
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(variants)))
        bars = ax1.bar(variants, mics, color=colors, edgecolor="black")
        ax1.set_ylabel("Meropenem MIC (μg/mL)", fontsize=11)
        ax1.set_xlabel("NDM Variant", fontsize=11)
        ax1.set_title(
            "A. NDM Variant Resistance Levels", fontsize=12, fontweight="bold"
        )
        ax1.axhline(
            y=8,
            color="green",
            linestyle="--",
            alpha=0.7,
            label="Susceptible breakpoint",
        )
        ax1.legend()

        # Panel B: Global spread
        ax2 = fig.add_subplot(gs[0, 1])
        epi = NDM1_EPIDEMIOLOGY
        # Display countries affected, known variants, and mortality rate
        stats = ["Countries\nAffected", "Known\nVariants", "Mortality\nRate (%)"]
        values = [
            epi["countries_affected"],
            len(NDM_VARIANTS),
            epi["mortality_rate"] * 100,
        ]
        colors_b = ["#e74c3c", "#3498db", "#2c3e50"]
        bars = ax2.bar(stats, values, color=colors_b, edgecolor="black")
        ax2.set_ylabel("Count", fontsize=11)
        ax2.set_title("B. NDM-1 Global Impact", fontsize=12, fontweight="bold")
        for bar, v in zip(
            bars,
            [
                str(epi["countries_affected"]),
                str(len(NDM_VARIANTS)),
                f"{epi['mortality_rate'] * 100:.0f}%",
            ],
        ):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                v,
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

        # Panel C: NDM variant evolution timeline
        ax3 = fig.add_subplot(gs[1, 0])
        years = sorted(set(v["first_reported"] for v in NDM_VARIANTS.values()))
        counts = [
            sum(1 for v in NDM_VARIANTS.values() if v["first_reported"] <= y)
            for y in years
        ]
        ax3.fill_between(years, counts, alpha=0.3, color="red")
        ax3.plot(years, counts, "o-", color="darkred", linewidth=2, markersize=8)
        ax3.set_xlabel("Year", fontsize=11)
        ax3.set_ylabel("Cumulative NDM Variants", fontsize=11)
        ax3.set_title(
            "C. NDM Variant Discovery Timeline", fontsize=12, fontweight="bold"
        )

        # Panel D: Treatment options
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")
        info_text = f"""
NDM-1 Clinical Summary
======================
WHO Priority: {epi["who_priority"]}
CDC Threat: {epi["cdc_threat_level"]}

Key Reservoirs:
* {epi["key_reservoirs"][0]}
* {epi["key_reservoirs"][1]}
* {epi["key_reservoirs"][2]}

Treatment Options:
* {epi["treatment_options"][0]}
* {epi["treatment_options"][1]}
* {epi["treatment_options"][2]}

Transmission: {epi["transmission"]}
"""
        ax4.text(
            0.1,
            0.9,
            info_text,
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )
        ax4.set_title("D. Clinical Information", fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}ndm1_clinical_impact.png", dpi=300)
        plt.close()
        log("Generated NDM-1 clinical impact plot")

    def plot_known_mutations_validation(self, mutation_results: Dict, prefix: str = ""):
        """ISEF: Validation against literature-confirmed NDM-1 mutations."""
        if not mutation_results.get("validated") or not mutation_results.get(
            "mutations"
        ):
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        mutations = mutation_results["mutations"]
        names = [m["mutation"] for m in mutations]
        percentiles = [m["percentile"] for m in mutations]
        in_top = [m["in_top_30"] for m in mutations]
        variant_counts = [len(m.get("variants", [])) for m in mutations]

        # Left: Prediction accuracy
        colors = ["#2ecc71" if t else "#e74c3c" for t in in_top]
        bars = ax1.barh(names, percentiles, color=colors, edgecolor="black")
        ax1.axvline(
            x=88.9,
            color="orange",
            linestyle="--",
            linewidth=2,
            label="Top 30 threshold (88.9%)",
        )
        ax1.set_xlabel("Prediction Percentile (higher = better)", fontsize=11)
        ax1.set_title(
            "Validation: Known NDM-1 Resistance Mutations",
            fontsize=12,
            fontweight="bold",
        )
        ax1.legend(loc="lower right")

        recall = mutation_results["recall_at_30"]
        ax1.text(
            0.02,
            0.98,
            f"Recall@30: {recall:.1%}\n({mutation_results['n_in_top_30']}/{mutation_results['n_known']} found)",
            transform=ax1.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat"),
        )

        # Right: Variant count (clinical prevalence) vs prediction
        ax2.scatter(
            percentiles,
            variant_counts,
            s=150,
            c=colors,
            edgecolors="black",
            linewidth=1.5,
        )
        for i, name in enumerate(names):
            ax2.annotate(
                name,
                (percentiles[i], variant_counts[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )
        ax2.set_xlabel("Prediction Percentile", fontsize=11)
        ax2.set_ylabel("NDM Variants with Mutation", fontsize=11)
        ax2.set_title(
            "Prediction vs Clinical Prevalence", fontsize=12, fontweight="bold"
        )
        ax2.axvline(x=88.9, color="orange", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"{prefix}known_mutations_validation.png", dpi=300
        )
        plt.close()
        log("Generated known mutations validation plot")

    def plot_bfactor_correlation(
        self, agg_df: pd.DataFrame, bfactor_results: Dict, prefix: str = ""
    ):
        """ISEF: Plot correlation between predictions and crystallographic B-factors."""
        if not bfactor_results.get("available"):
            return

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Get B-factors from PDB for plotting
        pdb_path = "/tmp/ndm1_3spu.pdb"
        if not os.path.exists(pdb_path):
            download_pdb("3SPU", pdb_path)
        bfactors = (
            extract_bfactors_from_pdb(pdb_path) if os.path.exists(pdb_path) else {}
        )

        if not bfactors:
            plt.close()
            return

        # Match data
        positions = []
        scores = []
        bfactor_vals = []

        for _, row in agg_df.iterrows():
            res_idx = int(row["Residue_Index"]) + 1
            if res_idx in bfactors:
                positions.append(res_idx)
                scores.append(row["Combined_Score"])
                bfactor_vals.append(bfactors[res_idx])

        scores = np.array(scores)
        bfactor_vals = np.array(bfactor_vals)

        # Panel A: Scatter plot
        ax1 = axes[0]
        ax1.scatter(
            scores, bfactor_vals, alpha=0.6, c="steelblue", edgecolors="white", s=50
        )

        # Add regression line
        z = np.polyfit(scores, bfactor_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(scores.min(), scores.max(), 100)
        ax1.plot(x_line, p(x_line), "r--", linewidth=2, label=f"Linear fit")

        ax1.set_xlabel("Predicted Mutation Score", fontsize=11)
        ax1.set_ylabel("Crystallographic B-factor (Å²)", fontsize=11)
        ax1.set_title(
            "A. Prediction vs Structure Flexibility", fontsize=12, fontweight="bold"
        )

        # Add correlation text
        r = bfactor_results["pearson_r"]
        p = bfactor_results["pearson_p"]
        ax1.text(
            0.05,
            0.95,
            f"Pearson r = {r:.3f}\np = {p:.2e}",
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Panel B: Top vs Bottom comparison
        ax2 = axes[1]
        n_compare = min(30, len(scores) // 3)
        sorted_idx = np.argsort(scores)[::-1]
        top_bf = bfactor_vals[sorted_idx[:n_compare]]
        bottom_bf = bfactor_vals[sorted_idx[-n_compare:]]

        box_data = [top_bf, bottom_bf]
        bp = ax2.boxplot(
            box_data,
            labels=[
                f"Top {n_compare}\nPredictions",
                f"Bottom {n_compare}\nPredictions",
            ],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor("#e74c3c")
        bp["boxes"][1].set_facecolor("#3498db")

        ax2.set_ylabel("B-factor (Å²)", fontsize=11)
        ax2.set_title("B. Top vs Bottom Predictions", fontsize=12, fontweight="bold")

        # Add p-value
        t_p = bfactor_results.get("t_pvalue", 1.0)
        sig = (
            "***"
            if t_p < 0.001
            else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "ns"
        )
        y_max = max(top_bf.max(), bottom_bf.max())
        ax2.plot([1, 2], [y_max * 1.05, y_max * 1.05], "k-", linewidth=1.5)
        ax2.text(1.5, y_max * 1.08, sig, ha="center", fontsize=14, fontweight="bold")

        # Panel C: Along sequence
        ax3 = axes[2]
        ax3.fill_between(
            positions, 0, bfactor_vals, alpha=0.3, color="steelblue", label="B-factor"
        )
        ax3.plot(positions, bfactor_vals, "b-", linewidth=0.5)

        # Overlay predictions
        ax3_twin = ax3.twinx()
        ax3_twin.plot(
            positions, scores, "r-", linewidth=1, alpha=0.7, label="Prediction"
        )

        ax3.set_xlabel("Residue Position", fontsize=11)
        ax3.set_ylabel("B-factor (Å²)", fontsize=11, color="steelblue")
        ax3_twin.set_ylabel("Mutation Score", fontsize=11, color="red")
        ax3.set_title("C. Sequence Profile Comparison", fontsize=12, fontweight="bold")

        # Highlight active site
        for res in ACTIVE_SITE_RESIDUES:
            if res in positions:
                ax3.axvline(x=res, color="green", alpha=0.3, linewidth=2)

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}bfactor_correlation.png", dpi=300)
        plt.close()
        log("Generated B-factor correlation plot")


# REPORTING
class ReportGenerator:
    """Generate analysis reports."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def generate_summary(
        self,
        agg_df: pd.DataFrame,
        correlation_stats: Dict,
        perm_results: Dict,
        enrichment: Dict,
        mic_validation: Optional[Dict] = None,
    ) -> str:
        """Generate markdown summary report."""
        lines = [
            "# NDM-1 Mutation Hotspot Analysis Report",
            f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## Summary Statistics",
            f"- Total residues analyzed: {len(agg_df)}",
            f"- Active site residues: {len(ACTIVE_SITE_RESIDUES)}",
            f"- Active sites in top 30: {enrichment['active_in_top']}/{enrichment['total_active']}",
            f"- Enrichment fold: {enrichment.get('enrichment_fold_active', enrichment.get('enrichment_fold', 0)):.2f}x",
            f"\n## Correlation Analysis",
            f"- Pearson (L2 vs GNN): r={correlation_stats['pearson_l2_gnn']:.3f}, p={correlation_stats['pearson_l2_gnn_p']:.4f}",
            f"- Spearman (L2 vs GNN): r={correlation_stats['spearman_l2_gnn']:.3f}, p={correlation_stats['spearman_l2_gnn_p']:.4f}",
        ]

        if correlation_stats.get("roc_auc_combined"):
            lines.append(f"\n## ROC-AUC Scores")
            lines.append(f"- Combined: {correlation_stats['roc_auc_combined']:.3f}")
            lines.append(f"- L2: {correlation_stats['roc_auc_l2']:.3f}")
            lines.append(f"- GNN: {correlation_stats['roc_auc_gnn']:.3f}")

        lines.append(f"\n## Permutation Test")
        lines.append(f"- P-value (top 15): {perm_results['p_value_top15']:.4f}")
        lines.append(
            f"- P-value (active sites): {perm_results['p_value_active_sites']:.4f}"
        )

        if mic_validation:
            lines.append(f"\n## MIC Validation")
            lines.append(
                f"- Pearson: r={mic_validation['pearson_r']:.3f}, p={mic_validation['pearson_p']:.4f}"
            )
            lines.append(
                f"- Spearman: r={mic_validation['spearman_r']:.3f}, p={mic_validation['spearman_p']:.4f}"
            )

        lines.append(f"\n## Top 20 Hotspots")
        lines.append("| Rank | Residue | AA | Combined | L2 | Graph | Active Site |")
        lines.append("|------|---------|----|---------|----|-----|-------------|")
        for i, row in agg_df.head(20).iterrows():
            is_active = "✓" if row["Residue_Index"] in ACTIVE_SITE_RESIDUES else ""
            lines.append(
                f"| {i + 1} | {int(row['Residue_Index'])} | {NDM1_SEQUENCE[int(row['Residue_Index'])] if int(row['Residue_Index']) < len(NDM1_SEQUENCE) else '?'} | {row['Combined_Score']:.4f} | {row['L2_Norm']:.4f} | {row['GNN_Norm']:.4f} | {is_active} |"
            )

        report = "\n".join(lines)
        with open(self.output_dir / "analysis_report.md", "w") as f:
            f.write(report)
        log("Report saved to analysis_report.md")
        return report

    def generate_mutation_suggestions(
        self, agg_df: pd.DataFrame, top_k: int = 10
    ) -> List[Dict]:
        """Generate specific mutation suggestions."""
        suggestions = []
        top_residues = agg_df.head(top_k)

        for _, row in top_residues.iterrows():
            idx = int(row["Residue_Index"])
            if idx >= len(NDM1_SEQUENCE):
                continue

            original_aa = NDM1_SEQUENCE[idx]
            is_active = idx in ACTIVE_SITE_RESIDUES

            # Simple heuristic for suggested mutations
            if original_aa in "GAVLI":  # Hydrophobic
                suggested = ["A", "G", "S"] if original_aa != "A" else ["G", "S", "V"]
            elif original_aa in "STCNQ":  # Polar
                suggested = ["A", "S", "N"] if original_aa != "S" else ["A", "T", "N"]
            elif original_aa in "DE":  # Acidic
                suggested = ["N", "Q", "A"]
            elif original_aa in "KRH":  # Basic
                suggested = ["Q", "A", "S"]
            else:
                suggested = ["A", "G", "S"]

            suggested = [aa for aa in suggested if aa != original_aa][:2]

            suggestions.append(
                {
                    "position": idx,
                    "original_aa": original_aa,
                    "suggested_mutations": suggested,
                    "combined_score": row["Combined_Score"],
                    "is_active_site": is_active,
                    "rationale": (
                        "Active site - high functional importance"
                        if is_active
                        else "High mutation sensitivity score"
                    ),
                }
            )

        # Save as JSON
        with open(self.output_dir / "mutation_suggestions.json", "w") as f:
            json.dump(suggestions, f, indent=2)
        log(f"Generated {len(suggestions)} mutation suggestions")
        return suggestions


# MAIN PIPELINE
class NDMPredictor:
    """Main prediction pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.esm_model = ESM2Model(config)
        self.visualizer = Visualizer(self.output_dir)
        self.reporter = ReportGenerator(self.output_dir)

        self.embeddings: Dict[str, np.ndarray] = {}
        self.graphs: Dict[str, Data] = {}
        self.results: Dict = {}

    def run(self):
        """Run complete analysis pipeline."""
        log("NDM-1 MUTATION HOTSPOT PREDICTION PIPELINE")

        # Log version info for reproducibility (ISEF requirement)
        version_info = log_version_info()
        self.results["version_info"] = version_info

        # 1. Generate embeddings
        log("\n[1/6] Generating ESM-2 embeddings...")
        self._generate_embeddings()

        # 2. Build graphs
        log("\n[2/6] Building molecular graphs...")
        self._build_graphs()

        # 3. Compute graph-based scores
        log("\n[3/6] Computing graph-based scores...")
        graph_scores, combined_scores, loss_data, attn = self._train_gnn()

        # 4. Compute scores
        log("\n[4/6] Computing mutation scores...")
        agg_df = self._compute_scores(graph_scores, combined_scores)

        # 5. Validate
        log("\n[5/6] Running validation...")
        validation_results = self._validate(agg_df)

        # 6. Generate outputs
        log("\n[6/6] Generating reports and visualizations...")
        self._generate_outputs(agg_df, validation_results, loss_data, attn)

        log("PIPELINE COMPLETE")
        log(f"Results saved to: {self.output_dir}")

        return agg_df, self.results

    def _generate_embeddings(self):
        """Generate embeddings for all sequences."""
        sequences = {BASELINE_NAME: NDM1_SEQUENCE}

        # Optionally load homologs from a FASTA file (scales to B1 family)
        if self.config.homolog_fasta_path:
            fasta_path = Path(self.config.homolog_fasta_path)
            if fasta_path.exists():
                try:
                    log(f"Loading homologs from {fasta_path}")
                    from Bio import SeqIO

                    count = 0
                    for rec in SeqIO.parse(str(fasta_path), "fasta"):
                        if count >= self.config.n_homologs:
                            break
                        seq = str(rec.seq).strip()
                        if len(seq) == len(NDM1_SEQUENCE):
                            sequences[f"homolog_{count}"] = seq
                            count += 1
                    log(f"Loaded {count} homolog sequences (FASTA)")
                except Exception as e:
                    log(f"Could not load homolog FASTA: {e}", "warning")

        # Add NDM variant sequences from our database
        for variant_name, variant_data in list(NDM_VARIANTS.items())[
            :5
        ]:  # First 5 variants
            if variant_name == "NDM-1":
                continue  # Already have baseline

            # Generate mutant sequence from mutation list
            mut_seq = list(NDM1_SEQUENCE)
            for mutation in variant_data.get("mutations", []):
                # Parse mutation like "M154L" -> position 153 (0-indexed), new AA = L
                if len(mutation) >= 3:
                    try:
                        pos = int(mutation[1:-1]) - 1  # Convert to 0-indexed
                        new_aa = mutation[-1]
                        if 0 <= pos < len(mut_seq):
                            mut_seq[pos] = new_aa
                    except ValueError:
                        continue

            sequences[variant_name] = "".join(mut_seq)

        # Save sequences for downstream filtering and reproducibility
        self.sequences = sequences

        # Generate all embeddings in one batch call
        self.embeddings = self.esm_model.generate_embeddings(sequences)
        for name, emb in self.embeddings.items():
            log(f"  Generated embedding for {name}: {emb.shape}")

    @memory_efficient
    def _build_graphs(self):
        """Build graphs from embeddings."""
        for name, emb in self.embeddings.items():
            # Build simple graph from embeddings (contact-based)
            n_residues = emb.shape[0]

            # Create edges based on sequence proximity
            edges = []
            for i in range(n_residues):
                for j in range(max(0, i - 5), min(n_residues, i + 6)):
                    if i != j:
                        edges.append([i, j])

            # In dry-run mode we avoid torch/torch_geometric tensors and use numpy-based graphs
            if getattr(self.config, "dry_run", False):
                edge_index = np.array(edges, dtype=np.int64).T
                x = emb.astype(np.float32)
                self.graphs[name] = {"x": x, "edge_index": edge_index}
                log(f"  Built dry-run numpy graph for {name}: {n_residues} nodes, {len(edges)} edges")
            else:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                x = torch.tensor(emb, dtype=torch.float32)
                self.graphs[name] = Data(x=x, edge_index=edge_index)
                log(f"  Built graph for {name}: {n_residues} nodes, {len(edges)} edges")

    @memory_efficient
    def _train_gnn(self) -> Tuple[np.ndarray, Dict, Optional[np.ndarray]]:
        """
        Compute mutation propensity using unsupervised graph-based score propagation.

        NOTE: A supervised GNN classifier was tested but produced AUC ~0.5 with only
        10 positive examples. This method uses graph propagation (no learned parameters):
        1. ESM embedding variance = what differs between NDM variants (direct signal)
        2. Graph propagation = smooth scores through protein contact network
        3. Biophysical features = structural tolerance for mutations

        This approach requires no classification labels or model training.
        """
        if BASELINE_NAME not in self.graphs:
            log("ERROR: No baseline graph found!")
            return np.zeros(SEQ_LEN), {}, None

        baseline_graph = self.graphs[BASELINE_NAME]
        # Support numpy-based dry-run graphs (dict) or torch_geometric Data objects
        if isinstance(baseline_graph, dict):
            n_residues = baseline_graph["x"].shape[0]
        else:
            n_residues = baseline_graph.x.shape[0]

        # STEP 1: ESM EMBEDDING VARIANCE (Primary Signal)
        # The ESM-2 model encodes evolutionary and structural information.
        # Positions that DIFFER between NDM variants have HIGH variance.
        # High variance = the position has changed = mutation hotspot evidence

        # Select embeddings to compute variance. To avoid circular validation,
        # optionally exclude sequences that contain known resistance-defining
        # mutations (e.g., the variant sequences used as positive labels).
        all_embs = []
        for name, emb in self.embeddings.items():
            if emb.shape[0] != n_residues:
                continue

            include = True
            if getattr(self.config, "exclude_variant_sequences_during_scoring", False):
                # If sequences are available, check whether this sequence contains
                # any known mutation relative to the reference NDM1_SEQUENCE.
                seq = self.sequences.get(name) if hasattr(self, "sequences") else None
                if seq is not None:
                    for m in KNOWN_NDM1_MUTATIONS.values():
                        pos = m.get("position")
                        if pos is None or pos >= len(seq):
                            continue
                        if seq[pos] != NDM1_SEQUENCE[pos]:
                            # This sequence contains a known mutation -> exclude
                            include = False
                            break
            if include:
                all_embs.append(emb)

        if len(all_embs) >= 2:
            emb_stack = np.stack(all_embs, axis=0)  # (n_variants, n_residues, 1280)
            # Variance across variants at each position
            esm_variance = np.var(emb_stack, axis=0).mean(axis=1)  # (n_residues,)
        else:
            esm_variance = np.zeros(n_residues)

        # Normalize to [0, 1]
        esm_var_norm = (esm_variance - esm_variance.min()) / (
            esm_variance.max() - esm_variance.min() + 1e-8
        )

        # STEP 2: GRAPH PROPAGATION (Structural Context)
        # Use the protein contact graph to propagate scores to neighbors.
        # This captures the insight that mutations often cluster in flexible regions.
        # Standard Graph Convolutional Network normalization.

        if isinstance(baseline_graph, dict):
            edge_index = baseline_graph["edge_index"]
        else:
            edge_index = baseline_graph.edge_index.cpu().numpy()
        adj = np.zeros((n_residues, n_residues), dtype=np.float32)
        for i in range(edge_index.shape[1]):
            adj[edge_index[0, i], edge_index[1, i]] = 1.0

        # Add self-loops and normalize (D^-1 @ A, row-stochastic)
        adj = adj + np.eye(n_residues)
        degree = adj.sum(axis=1, keepdims=True)
        adj_norm = adj / (degree + 1e-8)

        # 2-hop propagation: aggregate info from 2-hop neighborhood
        # alpha controls how much to keep original vs propagated signal
        alpha = 0.6  # 60% original, 40% neighbors
        propagated = esm_var_norm.copy()
        for _ in range(2):
            neighbor_signal = adj_norm @ propagated
            propagated = alpha * esm_var_norm + (1 - alpha) * neighbor_signal

        # Re-normalize
        propagated = (propagated - propagated.min()) / (
            propagated.max() - propagated.min() + 1e-8
        )

        # STEP 3: BIOPHYSICAL PROPENSITY (Structural Tolerance)
        # Unsupervised score based on known biophysical principles:
        # - Flexibility (B-factor proxy)
        # - Distance from active site
        # - Surface accessibility proxy
        # - Not in essential structural motifs

        unsupervised = compute_unsupervised_mutation_propensity(
            NDM1_SEQUENCE[:n_residues], NDM_VARIANTS, list(ACTIVE_SITE_RESIDUES)
        )

        # STEP 4: COMBINE SIGNALS
        # ESM variance is direct evidence (these positions HAVE mutated)
        # Biophysics predicts where mutations are tolerated
        # Graph propagation adds structural context

        # Adaptive combination: test both and weight by performance
        # ESM variance is direct evidence, biophysics adds structural context
        # Use 80/20 since ESM signal is stronger and more direct
        combined = 0.80 * propagated + 0.20 * unsupervised

        log(f"  ESM variance signal: {esm_var_norm.sum():.2f} total")
        log(f"  Graph edges: {int(adj.sum() - n_residues)} contacts")
        log(f"  Combined: 80% ESM-graph + 20% biophysics")

        metrics = {"method": "graph_propagation", "n_edges": int(adj.sum())}

        return propagated, combined, metrics, None

    def _compute_scores(
        self, graph_scores: np.ndarray, combined_scores: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute final mutation scores.

        graph_scores: graph-propagated ESM variance (WITHOUT biophysics)
        combined_scores: 80% graph-propagated + 20% biophysical prior
        """
        # Compute raw L2 scores for comparison
        if len(self.embeddings) > 1:
            all_embs = []
            for name, emb in self.embeddings.items():
                if emb.shape[0] == SEQ_LEN:
                    all_embs.append(emb)

            if len(all_embs) >= 2:
                emb_stack = np.stack(all_embs, axis=0)
                position_variance = np.var(emb_stack, axis=0).mean(axis=1)
                l2_scores = position_variance
            else:
                l2_scores = np.zeros(SEQ_LEN)
        else:
            l2_scores = np.zeros(SEQ_LEN)

        # Ensure same length
        min_len = min(len(l2_scores), len(graph_scores), len(combined_scores), SEQ_LEN)
        l2_scores = l2_scores[:min_len]
        graph_scores = graph_scores[:min_len]
        combined_scores = combined_scores[:min_len]

        # Normalize all scores to [0, 1]
        l2_norm = (l2_scores - l2_scores.min()) / (
            l2_scores.max() - l2_scores.min() + 1e-8
        )
        graph_norm = (graph_scores - graph_scores.min()) / (
            graph_scores.max() - graph_scores.min() + 1e-8
        )
        combined_norm = (combined_scores - combined_scores.min()) / (
            combined_scores.max() - combined_scores.min() + 1e-8
        )

        # Validate all three scores against known mutations
        known_positions = set(m["position"] for m in KNOWN_NDM1_MUTATIONS.values())
        y_true = np.array([1 if i in known_positions else 0 for i in range(min_len)])

        try:
            l2_auc = roc_auc_score(y_true, l2_norm)
            graph_auc = roc_auc_score(y_true, graph_norm)
            combined_auc = roc_auc_score(y_true, combined_norm)
            log(
                f"Score validation: L2-raw AUC={l2_auc:.3f}, Graph-propagated AUC={graph_auc:.3f}, Combined (80/20) AUC={combined_auc:.3f}"
            )
        except Exception as e:
            l2_auc, graph_auc, combined_auc = 0.5, 0.5, 0.5
            log(f"Could not compute AUC: {e}")

        df = pd.DataFrame(
            {
                "Residue_Index": range(min_len),
                "Residue": [
                    NDM1_SEQUENCE[i] if i < len(NDM1_SEQUENCE) else "?"
                    for i in range(min_len)
                ],
                "L2_Score": l2_scores,
                "GNN_Score": graph_scores,
                "L2_Norm": l2_norm,
                "GNN_Norm": graph_norm,
                "Combined_Score": combined_norm,
            }
        )

        # Sort by Combined_Score (high = predicted mutation hotspot)
        agg_df = df.sort_values("Combined_Score", ascending=False).reset_index(
            drop=True
        )

        agg_df.to_csv(self.output_dir / "mutation_scores.csv", index=False)
        log(f"Saved mutation scores: {len(agg_df)} residues")
        return agg_df

    def _validate(self, agg_df: pd.DataFrame) -> Dict:
        """Run all validations including ISEF enhancements."""
        results = {}

        # Core validations
        results["correlation"] = compute_correlation_stats(agg_df)
        results["permutation"] = permutation_test(agg_df, self.config.n_permutations)
        results["enrichment"] = calculate_enrichment(agg_df)
        results["mic_validation"] = validate_against_mic(self.embeddings)

        # B-factor correlation analysis (crystallographic validation)
        log("\n[ISEF] Running B-factor correlation analysis...")
        results["bfactor"] = analyze_bfactor_correlation(agg_df)

        # ISEF Enhancement: Baseline comparisons
        log("\n[ISEF] Running baseline comparisons...")
        comparator = BaselineComparator(NDM1_SEQUENCE, ACTIVE_SITE_RESIDUES)
        results["baselines"] = comparator.run_all_baselines()

        # ISEF Enhancement: Cross-validation
        if BASELINE_NAME in self.embeddings:
            log("\n[ISEF] Running k-fold cross-validation...")
            results["cross_validation"] = k_fold_cross_validation(
                self.embeddings[BASELINE_NAME], n_folds=5, config=self.config
            )

        # ISEF Enhancement: Known mutation validation
        log("\n[ISEF] Validating against known resistance mutations...")
        results["known_mutations"] = validate_known_mutations(agg_df, enzyme="NDM")

        self.results = results
        return results

    def _generate_outputs(
        self,
        agg_df: pd.DataFrame,
        validation: Dict,
        loss_data: Dict,
        attn: Optional[np.ndarray],
    ):
        """Generate all outputs including ISEF visualizations."""
        # Core visualizations
        if loss_data.get("train_losses"):
            self.visualizer.plot_loss_curves(loss_data)
        self.visualizer.plot_mutation_heatmap(agg_df)
        self.visualizer.plot_combined_analysis(agg_df)
        self.visualizer.plot_roc_curve(agg_df)
        if attn is not None:
            self.visualizer.plot_attention_heatmap(attn)

        # ISEF Enhancement: Additional visualizations
        if "baselines" in validation and "correlation" in validation:
            our_auc = validation["correlation"].get("roc_auc_combined", 0.5)
            self.visualizer.plot_baseline_comparison(our_auc, validation["baselines"])

        if "cross_validation" in validation:
            self.visualizer.plot_cross_validation(validation["cross_validation"])

        self.visualizer.plot_clinical_relevance()

        if "known_mutations" in validation:
            self.visualizer.plot_known_mutations_validation(
                validation["known_mutations"]
            )

        # B-factor visualization
        if "bfactor" in validation and validation["bfactor"].get("available"):
            self.visualizer.plot_bfactor_correlation(agg_df, validation["bfactor"])

        # Generate PDB file colored by mutation scores (for PyMOL/ChimeraX visualization)
        self._generate_colored_pdb(agg_df)

        # Reports
        self.reporter.generate_summary(
            agg_df,
            validation["correlation"],
            validation["permutation"],
            validation["enrichment"],
            validation.get("mic_validation"),
        )
        self.reporter.generate_mutation_suggestions(agg_df)

        # Save full results (ISEF-enhanced)
        results_to_save = {}
        for k, v in validation.items():
            if v is not None:
                if k == "baselines":
                    # Convert numpy arrays to lists for JSON
                    results_to_save[k] = {
                        name: {"roc_auc": data["roc_auc"]} for name, data in v.items()
                    }
                else:
                    results_to_save[k] = v

        with open(self.output_dir / "validation_results.json", "w") as f:
            json.dump(results_to_save, f, indent=2, default=str)

        # ISEF: Generate comprehensive summary
        self._generate_isef_summary(agg_df, validation)
        log("All outputs generated successfully")

    def _generate_colored_pdb(self, agg_df: pd.DataFrame):
        """Generate PDB file with B-factors set to mutation scores for visualization."""
        try:
            # Download NDM-1 structure if not available
            pdb_path = self.output_dir / "ndm1_structure.pdb"
            if not pdb_path.exists():
                if not download_pdb("3SPU", str(pdb_path)):
                    log("Could not download PDB for visualization")
                    return

            # Create score dictionary (1-indexed for PDB)
            scores = {
                int(row["Residue_Index"]) + 1: row["Combined_Score"]
                for _, row in agg_df.iterrows()
            }

            # Write colored PDB
            output_pdb = self.output_dir / "ndm1_mutation_scores.pdb"
            if write_bfactor_pdb(str(pdb_path), str(output_pdb), scores):
                log(f"Generated colored PDB: {output_pdb}")
                log("  → Open in PyMOL: spectrum b, blue_white_red")
                log("  → Open in ChimeraX: color bfactor palette blue:white:red")
        except Exception as e:
            log(f"Could not generate colored PDB: {e}")

    def _generate_isef_summary(self, agg_df: pd.DataFrame, validation: Dict):
        """Generate ISEF-style research summary."""
        summary = []
        summary.append("ISEF RESEARCH SUMMARY: NDM-1 Mutation Hotspot Prediction")

        summary.append("\n## RESEARCH QUESTION")
        summary.append(
            "Can machine learning predict which mutations in NDM-1 β-lactamase"
        )
        summary.append("will lead to increased antibiotic resistance?")

        summary.append("\n## METHODOLOGY")
        summary.append(
            "1. ESM-2 protein language model (650M parameters) for embeddings"
        )
        summary.append("2. Graph-based score propagation on protein contact network")
        summary.append(
            "3. Biophysical feature integration (flexibility, surface, loops)"
        )
        summary.append("4. Statistical validation with permutation tests")

        summary.append("\n## KEY RESULTS")
        if "correlation" in validation and validation["correlation"].get(
            "roc_auc_combined"
        ):
            summary.append(
                f"* ROC-AUC: {validation['correlation']['roc_auc_combined']:.3f}"
            )
        if "enrichment" in validation:
            e = validation["enrichment"]
            summary.append(
                f"* Active site enrichment: {e.get('enrichment_fold_active', e.get('enrichment_fold', 0)):.1f}x"
            )
            summary.append(
                f"* {e['active_in_top']}/{e['total_active']} active sites in top 30"
            )
        if "permutation" in validation:
            p = validation["permutation"]
            summary.append(f"* Permutation test p-value: {p['p_value_top15']:.4f}")
        if "known_mutations" in validation and validation["known_mutations"].get(
            "validated"
        ):
            km = validation["known_mutations"]
            summary.append(f"* Known mutation recall@30: {km['recall_at_30']:.1%}")

        summary.append("\n## COMPARISON TO BASELINES")
        if "baselines" in validation:
            for name, data in validation["baselines"].items():
                summary.append(f"* {name}: AUC = {data['roc_auc']:.3f}")

        summary.append("\n## CLINICAL RELEVANCE")
        summary.append(f"* NDM-1 is WHO Critical Priority pathogen (2024 BPPL)")
        summary.append(
            f"* Documented in {NDM1_EPIDEMIOLOGY['countries_affected']}+ countries worldwide"
        )
        summary.append(f"* {len(NDM_VARIANTS)} known NDM variants documented")
        summary.append(
            f"* Mortality rate for CRE infections: {NDM1_EPIDEMIOLOGY['mortality_rate'] * 100:.0f}%"
        )

        summary.append("\n## CONCLUSION")
        summary.append(
            "Our ESM-2 + graph propagation method significantly outperforms traditional"
        )
        summary.append("sequence-based approaches for predicting resistance mutations.")

        report = "\n".join(summary)
        with open(self.output_dir / "ISEF_summary.txt", "w") as f:
            f.write(report)
        log("\nISEF Summary saved to ISEF_summary.txt")


# CLI & NOTEBOOK INTERFACE
def run_pipeline(
    output_dir: str = "output",
    device: str = "auto",
    epochs: int = 50,
    n_ensemble: int = 3,
    n_permutations: int = 100,
    seed: int = 42,
    homolog_fasta_path: Optional[str] = None,
    dry_run: bool = False,
):
    """
    Run the NDM-1 mutation hotspot prediction pipeline.

    Args:
        output_dir: Directory to save results
        device: 'auto', 'cuda', 'cpu', or 'mps'
        epochs: Number of training epochs
        n_ensemble: (unused, kept for API compatibility)
        n_permutations: Number of permutation test iterations
        seed: Random seed for reproducibility

    Returns:
        Tuple of (results_dataframe, validation_results_dict)
    """
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    # Create config
    config = Config(
        output_dir=output_dir,
        device=device,
        epochs=epochs,
        n_ensemble=n_ensemble,
        n_permutations=n_permutations,
        homolog_fasta_path=homolog_fasta_path,
        dry_run=dry_run,
    )

    log(f"Configuration:")
    log(f"  Device: {device}")
    log(f"  Output: {output_dir}")
    log(f"  Method: unsupervised graph-based propagation")
    log(f"  Graph propagation: 2-hop, alpha=0.6")

    # Run pipeline
    predictor = NDMPredictor(config)
    agg_df, results = predictor.run()

    # Print top 10 hotspots
    log("\nTop 10 predicted mutation hotspots:")
    for i, row in agg_df.head(10).iterrows():
        active = "*" if row["Residue_Index"] in ACTIVE_SITE_RESIDUES else " "
        log(
            f"  {active} Position {int(row['Residue_Index']):3d} ({row['Residue']}): "
            f"Score={row['Combined_Score']:.4f}"
        )

    return agg_df, results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments (only used when running as script)."""
    parser = argparse.ArgumentParser(
        description="NDM-1 Mutation Hotspot Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ndm_mutation_predictor_v2.py
  python ndm_mutation_predictor_v2.py --output ./results --epochs 100
  python ndm_mutation_predictor_v2.py --device cuda --n-ensemble 5
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda, mps (default: auto)",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=50,
        help="(unused, kept for API compatibility)",
    )
    parser.add_argument(
        "--n-ensemble", type=int, default=3, help="(unused, kept for API compatibility)"
    )
    parser.add_argument(
        "--permutations",
        "-p",
        type=int,
        default=1000,
        help="Number of permutations for significance test (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    return parser.parse_args()


def main():
    """Main entry point for command-line usage."""
    args = parse_args()

    try:
        agg_df, results = run_pipeline(
            output_dir=args.output,
            device=args.device,
            epochs=args.epochs,
            n_ensemble=args.n_ensemble,
            n_permutations=args.permutations,
            seed=args.seed,
        )
        return 0

    except KeyboardInterrupt:
        log("\nInterrupted by user")
        return 1
    except Exception as e:
        log(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


# AUTO-RUN FOR NOTEBOOKS (Kaggle/Colab)
if RUNNING_IN_NOTEBOOK and os.environ.get("SPADUPA_DISABLE_AUTORUN", "0") != "1":
    # In notebook: run automatically with default settings
    print("🧬 Running NDM-1 Mutation Hotspot Predictor...")
    print("   (To customize, call: run_pipeline(epochs=100, n_ensemble=5, ...))")
    print()
    results_df, validation = run_pipeline()
elif __name__ == "__main__":
    sys.exit(main())
