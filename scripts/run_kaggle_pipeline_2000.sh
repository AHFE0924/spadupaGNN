#!/usr/bin/env bash
set -euo pipefail

# Fast Kaggle pipeline tuned for ~2000 sequences on T4 x2.
# Usage: bash scripts/run_kaggle_pipeline_2000.sh

cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd)
export PYTHONPATH=${REPO_ROOT}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1

# Tunable defaults for a large but still Kaggle-friendly run
DATASET_MAX_SEQS=2000
EMBED_BATCH_SIZE=32
GKFOLD_BATCH_SIZE=8
CLUSTER_IDENTITY=0.4
GKFOLD_IDENTITY=0.3
MAX_FAMILY_SEQS=2000
OUTPUT_ROOT=output/kaggle_2000_fast
CLUSTER_PREFIX=output/clusters/b1_superfamily_40
EMBED_CACHE=output/embeddings_cache_2000.npz

mkdir -p output/clusters output/groupkfold "$OUTPUT_ROOT"

# Update repo
git pull

# Install Python deps
python -m pip install -q --upgrade pip
python -m pip install -q fair-esm biopython scikit-learn pandas numpy scipy

# Install cd-hit for fast clustering
apt-get update && apt-get install -y cd-hit

# Fetch + filter + dedup + cluster B1 superfamily
# Cap the final FASTA to the first 2000 filtered sequences for a bounded run.
python scripts/fetch_b1_superfamily.py \
  --output data/b1_superfamily.fasta \
  --raw-output data/b1_superfamily_raw.fasta \
  --taxon 2 \
  --min-length 200 --max-length 350 \
  --exclude-fragments \
  --max-ambiguous 5 \
  --dedup \
  --max-seqs "${DATASET_MAX_SEQS}" \
  --cluster-identity "${CLUSTER_IDENTITY}" \
  --cluster-output "${CLUSTER_PREFIX}" \
  --split-dir data/b1_families \
  --stats-output output/b1_superfamily_stats.json \
  --method auto

# Leakage-free GroupKFold for IMP and VIM
python scripts/groupkfold_cv.py \
  --input data/b1_superfamily.fasta \
  --clusters "${CLUSTER_PREFIX}.csv" \
  --family IMP \
  --identity "${GKFOLD_IDENTITY}" \
  --device cuda \
  --batch-size "${GKFOLD_BATCH_SIZE}" \
  --output output/groupkfold/imp

python scripts/groupkfold_cv.py \
  --input data/b1_superfamily.fasta \
  --clusters "${CLUSTER_PREFIX}.csv" \
  --family VIM \
  --identity "${GKFOLD_IDENTITY}" \
  --device cuda \
  --batch-size "${GKFOLD_BATCH_SIZE}" \
  --output output/groupkfold/vim

# Optimized full-family run with AMP + cache + DataParallel
python scripts/kaggle_multi_enzyme_real.py \
  --input data/b1_superfamily.fasta \
  --device cuda \
  --batch-size "${EMBED_BATCH_SIZE}" \
  --embed-cache "${EMBED_CACHE}" \
  --amp \
  --data-parallel \
  --profile \
  --max-family-seqs "${MAX_FAMILY_SEQS}" \
  --output "${OUTPUT_ROOT}"

echo "Done. Outputs:"
echo "- output/b1_superfamily_stats.json"
echo "- output/groupkfold/imp/enzyme_auc_summary.csv"
echo "- output/groupkfold/vim/enzyme_auc_summary.csv"
echo "- ${OUTPUT_ROOT}/enzyme_auc_summary.csv"
