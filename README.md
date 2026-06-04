# GNN
This project predicts mutation-tolerant positions in B1 metallo-beta-lactamases using ESM-2 embeddings and graph-based propagation.

## Superfamily data expansion
Fetch B1 MBL sequences from UniProt and cluster at 40% identity:

```bash
python scripts/fetch_b1_superfamily.py --output data/b1_superfamily.fasta --cluster-identity 0.4 --cluster-output output/clusters/b1_superfamily_40
```

## GroupKFold evaluation (no leakage)
Run GroupKFold CV with ROC/PR curves and mean/std AUC:

```bash
python scripts/groupkfold_cv.py --input data/b1_filtered.fasta --family VIM --identity 0.4 --cluster-method auto --output output/groupkfold --device cuda --folds 5
```

## In silico mutational heatmap
Generate all single amino-acid substitutions for a reference sequence:

```bash
python scripts/mutational_heatmap.py --output output/heatmap --device cuda
```

## External DMS validation
Validate predictions against a DMS dataset:

```bash
python scripts/dms_external_validation.py --dms path/to/dms.csv --predictions output/ndm1_mutation_scores.csv --mutation-column mutation --dms-score score
```

## Synthetic validation suite
Generate controlled synthetic datasets with planted hotspots and evaluate recovery:

```bash
# generate one dataset
python scripts/simulate_superfamily.py --output-dir output/synthetic/example --seqs-per-family 20 --length 250 --motif HXD

# evaluate with mock embeddings (fast) or real ESM embeddings
python scripts/evaluate_synthetic.py --fasta output/synthetic/example/synthetic.fasta --labels output/synthetic/example/labels.json --output output/synthetic/example_eval --mock-embeddings

# run the full multi-scenario suite
python scripts/run_synthetic_suite.py --output-dir output/synthetic_suite --mock-embeddings
```
