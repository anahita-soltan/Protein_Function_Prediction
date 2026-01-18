# Protein Function Prediction (GO Annotation)

This repository contains a reproducible pipeline for protein function prediction using Gene Ontology (GO) annotations.
The model predicts GO terms independently for the three sub-ontologies:

- Molecular Function (MF)
- Biological Process (BP)
- Cellular Component (CC)

Predictions are evaluated using the CAFA evaluator, and performance is compared against Naive and InterPro baselines.

## Repository structure


**Core scripts:**

- train.py — Trains one-vs-rest logistic regression models for MF, BP, and CC
- predict.py — Generates test-set predictions and final submission file
- cafa_val.py — Hold-out validation and CAFA-style evaluation

**Configuration:**

- config.yaml — All paths, hyperparameters, random seeds


**Utilities** (utils/)

- io.py — Data loading and saving
- labels.py — GO label construction and binarization
- blast.py — BLAST-based score integration
- submit.py — Submission formatting and constraint handling


**Outputs** (output/)

- models/ — Trained models (.joblib)
- preds/ — Test-set predictions
- cafa_val/ — Validation predictions and ground truth
- submission.tsv — Final submission file


**Notebooks**

- notebooks/ — Colab and analysis notebooks

**Metadata**

- requirements.txt

README.md



## Reproducibility
All randomness controlled via config.yaml
Training, validation, and prediction are deterministic
Models and outputs can be regenerated from scratch using the provided scripts.

## Requirements
Python ≥ 3.9
Linux / macOS (tested)
~16–32 GB RAM recommended for large subsets

## Python dependencies
pip install -r requirements.txt
pip install cafaeval

### Core libraries used:
- numpy, pandas
- scikit-learn
- joblib
- h5py
- matplotlib
- cafaeval (official CAFA evaluator)

## Data setup
The project expects the following directory structure (paths configurable):

biological_data_pfp/
─ train/
─ train_set.tsv
─ train_ids.txt
─ train_embeddings.h5
─ train_protein2ipr.dat
─ go-basic.obo
├─ test/
─ test_ids.txt
─ test_embeddings.h5
─ test_protein2ipr.dat
─ blast_test_results.tsv

Update paths in config.yaml accordingly

## Configuration (config.yaml)
All paths and hyperparameters are controlled from one file. Key options:

data:
  train_dir: "/path/to/train"
  test_dir:  "/path/to/test"
  out_dir:   "/path/to/output"
model:
  subset_n: 20000      # null = use all training proteins
  val_frac: 0.1        # validation split for CAFA-style evaluation
  random_state: 42
  max_iter: 200
blast:
  enabled: true
  evalue_thresh: 0.001
  topk: 50
  alpha: 0.2           # final score = alpha*ML + (1-alpha)*BLAST
predict:
  topk_per_aspect: 500
  min_score: 1e-6

## Training the models
Trains three independent OvR logistic regression models (MF, BP, CC).
**python train.py --config config.yaml**

### Outputs:
output/models/
- model_mf.joblib
- model_bp.joblib
─ model_cc.joblib**
Training is fully reproducible given the same config and random seed.

## Predicting on the test set
Generates the final submission file:
**python predict.py --config config.yaml**

Output: **output/submission.tsv**
Format (CAFA-compliant):
Protein_ID   GO_term     score
P12345       GO:0008150  0.742
P12345       GO:0003674  0.531
...
Each protein is capped at 1500 GO terms total across aspects.

## Internal validation (CAFA-style)
To evaluate performance on a held-out subset of the training data:
**python cafa_val.py**

This will:
Split training data into train/validation
Generate CAFA-formatted predictions
Save:
output/cafa_val/
├── ground_truth_val.tsv
├── ours_val.tsv
└── preds/

### Run CAFA evaluation
cafaeval \
  path/to/go-basic.obo \
  output/cafa_val \
  output/cafa_val/ground_truth_val.tsv \
  -out_dir output/cafa_val_results

Produces:
**evaluation_all.tsv
evaluation_best_f.tsv
Precision-recall curves and figures**




