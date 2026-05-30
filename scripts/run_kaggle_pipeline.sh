#!/usr/bin/env bash
set -euo pipefail

# Full Kaggle pipeline runner
# Usage: bash scripts/run_kaggle_pipeline.sh

cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd)
export PYTHONPATH=${REPO_ROOT}

# Update repo
git pull

# Install Python deps
python -m pip install -q --upgrade pip
python -m pip install -q fair-esm biopython scikit-learn pandas numpy scipy

# Install cd-hit for clustering
apt-get update && apt-get install -y cd-hit

# Fetch and cluster B1 superfamily (filtered + dedup)
python scripts/fetch_b1_superfamily.py \
  --output data/b1_superfamily.fasta \
  --raw-output data/b1_superfamily_raw.fasta \
  --taxon 2 \
  --min-length 200 --max-length 350 \
  --exclude-fragments \
  --max-ambiguous 5 \
  --dedup \
  --cluster-identity 0.4 \
  --cluster-output output/clusters/b1_superfamily_40 \
  --split-dir data/b1_families \
  --stats-output output/b1_superfamily_stats.json \
  --method auto

# Leakage-free GroupKFold for IMP (example)
python scripts/groupkfold_cv.py \
  --input data/b1_superfamily.fasta \
  --clusters output/clusters/b1_superfamily_40.csv \
  --family IMP \
  --identity 0.3 \
  --device cuda \
  --batch-size 8 \
  --output output/groupkfold/imp

# Optimized full family run (ESM-2 with AMP, embed cache, DataParallel)
python scripts/kaggle_multi_enzyme_real.py \
  --input data/b1_superfamily.fasta \
  --device cuda \
  --batch-size 32 \
  --embed-cache output/embeddings_cache.npz \
  --amp \
  --data-parallel \
  --profile \
  --max-family-seqs 2000 \
  --output output/kaggle_real_fast

echo "All steps completed. Check output/ for results and output/b1_superfamily_stats.json for dataset stats."
