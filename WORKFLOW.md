# FIRE 2019 AILA Legal Retrieval Workflow

This document explains the workflow used to build and evaluate the legal retrieval
pipeline in this project. The goal was to create an end-to-end local system for
the FIRE 2019 AILA dataset that can retrieve relevant case laws and statutes for
legal queries, evaluate the retrieval quality, and compare multiple retrieval
approaches.

## 1. Project Setup

The project was organized into separate modules for data loading, preprocessing,
retrieval, evaluation, experiments, and dashboarding.

Main files and folders:

```text
main.py
data_loader.py
preprocessing.py
retrieval/
  tfidf.py
  bm25.py
  embeddings.py
  statute_classifier.py
evaluation/
  metrics.py
dashboard/
  app.py
experiments.py
tune_fusion.py
data/
models/
outputs/
```

The dependencies were listed in `requirements.txt`:

```text
pandas
numpy
scikit-learn
sentence-transformers
rank_bm25
streamlit
matplotlib
kagglehub
joblib
```

The `.gitignore` file was added so that large generated files, downloaded data,
cached models, outputs, Python caches, and virtual environments are not committed.

## 2. Dataset Download And Preparation

The pipeline uses `kagglehub` to download the FIRE 2019 AILA dataset. The data
loader prepares the local data structure under `data/` and separates the dataset
into queries, case documents, statute documents, and relevance judgments.

The prepared data layout is:

```text
data/
  queries/
  cases/
  statutes/
  qrels/
  processed/
```

The raw text files are parsed into structured tables and saved under
`data/processed/` so later runs can reuse the processed data without repeating
the full parsing step.

## 3. Text Preprocessing

The pipeline adds a `clean_text` column to queries, cases, and statutes. This
cleaned text is used by the retrieval models.

The preprocessing step handles basic normalization such as lowercasing and text
cleanup. The pipeline also supports an optional stopword removal flag:

```bash
python main.py --remove-stopwords
```

The processed CSV files are saved as:

```text
data/processed/queries.csv
data/processed/cases.csv
data/processed/statutes.csv
```

## 4. Baseline Retrieval Pipeline

The main pipeline is run with:

```bash
python main.py --top-k 10
```

The baseline retrieval systems are:

1. Case retrieval with TF-IDF.
2. Case retrieval with BM25.
3. Case retrieval with sentence embeddings.
4. Statute retrieval with TF-IDF.
5. Statute retrieval with sentence embeddings.
6. Statute retrieval with a supervised multi-label classifier when qrels are
   available.

The statute classifier follows the AILA split used in this project:

```text
Training queries: AILA_Q1 to AILA_Q10
Evaluation queries: AILA_Q11 to AILA_Q50
```

The classifier is trained only on the early training queries, while metrics are
reported on later evaluation queries.

## 5. Model Caching

The project saves reusable model artifacts under `models/`.

Examples include:

```text
models/case_tfidf_tfidf.joblib
models/case_bm25_bm25.joblib
models/case_embeddings_embeddings.npy
models/statute_classifier.joblib
```

Sentence embedding files are cached because embedding all documents can be slow.
The `--force` flag can be used to rebuild cached data and embeddings:

```bash
python main.py --top-k 10 --force
```

One known review note is that the embedding cache currently checks document IDs,
but does not fully validate preprocessing changes, model name, or truncation
settings before reuse. Use `--force` when changing preprocessing or embedding
settings.

## 6. Evaluation

Each retrieval run is evaluated against qrels using standard information
retrieval metrics:

```text
Precision@K
Recall@K
F1@K
MAP
nDCG@K
```

The evaluation outputs are saved under `outputs/`.

Important files:

```text
outputs/metrics_summary.csv
outputs/metrics_summary.json
outputs/*_per_query_metrics.csv
outputs/*_results.csv
outputs/*.trec
```

The `.csv` files are useful for analysis in Python or spreadsheets. The `.trec`
files are saved in standard TREC run format.

## 7. Experiment Sweeps

Extra experiments were added in `experiments.py` to compare stronger retrieval
configurations.

Fast sweep:

```bash
python experiments.py --mode fast --top-k 10
```

Strong sweep:

```bash
python experiments.py --mode strong --top-k 10
```

The experiment script tests:

1. Additional sentence-transformer models.
2. TF-IDF and BM25 fusion.
3. Lexical plus embedding fusion.
4. Classifier-based statute fusion.
5. Cross-encoder reranking.
6. Extra binary rankers for statute retrieval.

Experiment outputs are saved under:

```text
outputs/experiments/
```

Important leaderboard files:

```text
outputs/experiments/leaderboard_fast.csv
outputs/experiments/leaderboard_strong.csv
```

## 8. Fusion Tuning

The `tune_fusion.py` script performs a grid search over retrieval fusion weights.
It combines different retrieval signals and selects the best scoring weighted
combination.

Run:

```bash
python tune_fusion.py
```

The tuning script evaluates combinations such as:

```text
case: TF-IDF + BM25 + MiniLM embeddings + MPNet embeddings
statute: classifier + TF-IDF + BM25 + MiniLM embeddings + MPNet embeddings
```

The best tuned outputs are:

```text
outputs/experiments/case_best_tuned.csv
outputs/experiments/case_best_tuned.trec
outputs/experiments/statute_best_tuned.csv
outputs/experiments/statute_best_tuned.trec
outputs/experiments/best_tuned_summary.csv
```

Current saved best tuned summary:

```text
case MAP:    0.1502281746031746
statute MAP: 0.19247123015873016
```

One review note is that fusion tuning currently optimizes and reports on the same
Q11-Q50 evaluation split. That is acceptable for exploratory comparison, but a
separate validation/test split would be better for final reporting.

## 9. Dashboard

A Streamlit dashboard was added under `dashboard/app.py`.

Run:

```bash
streamlit run dashboard/app.py
```

The dashboard is intended to make the results easier to inspect. It includes
model comparison views, metric tables, best-model summaries, and an interactive
query explorer for case and statute retrieval outputs.

## 10. Final Output Flow

The complete workflow is:

```text
Install requirements
        |
        v
Download / prepare FIRE 2019 AILA dataset
        |
        v
Parse queries, cases, statutes, and qrels
        |
        v
Preprocess text and save processed CSV files
        |
        v
Train or load retrieval models
        |
        v
Generate ranked results for cases and statutes
        |
        v
Evaluate with Precision, Recall, F1, MAP, and nDCG
        |
        v
Save CSV, JSON, and TREC outputs
        |
        v
Run extra experiments and fusion tuning
        |
        v
Inspect results in leaderboard files or Streamlit dashboard
```

## 11. Recommended Commands

Basic setup:

```bash
pip install -r requirements.txt
```

Run the main pipeline:

```bash
python main.py --top-k 10
```

Rebuild cached artifacts:

```bash
python main.py --top-k 10 --force
```

Run stronger experiments:

```bash
python experiments.py --mode strong --top-k 10
```

Tune fusion weights:

```bash
python tune_fusion.py
```

Open dashboard:

```bash
streamlit run dashboard/app.py
```

## 12. What Was Achieved

The final project contains a complete local legal information retrieval system
for the AILA dataset. It can prepare the dataset, build multiple retrieval
models, evaluate them with standard IR metrics, save ranked outputs, run stronger
experimental comparisons, tune fusion weights, and visualize results through a
dashboard.

The strongest saved tuned runs are available in:

```text
outputs/experiments/case_best_tuned.csv
outputs/experiments/statute_best_tuned.csv
```

These files represent the best current ranked outputs produced by the workflow.
