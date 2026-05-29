# Research Steps Log

## 2026-05-28
- Removed ISEF-specific wording and emojis from the pipeline for a cleaner, professional tone.
- Added sequence clustering utilities with cd-hit support and a Biopython fallback for 30% identity clustering.
- Added GroupKFold cross-validation script with ROC/PR curves and mean/std AUC reporting.
- Added residue-importance script to map embedding-based importance scores onto NDM-1 structure.
- Fixed script import paths for Kaggle execution (groupkfold_cv, cluster_sequences, residue_importance).
- Added GroupKFold guardrails for small cluster counts (write summary and exit cleanly).

## 2026-05-29
- Added UniProt superfamily fetch + clustering script (B1 MBL families, 40% identity).
- Added GroupKFold CV permutation/CI reporting for mean ROC/PR AUC stability.
- Added in silico mutational heatmap generator for single-site substitutions.
- Added external DMS validation script for leakage-free benchmarking.
- Added residue-importance permutation option for embedding-based interpretation.
