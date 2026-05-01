# FIRE 2019 AILA Legal Retrieval Pipeline

End-to-end local pipeline for the FIRE 2019 AILA dataset:

- downloads the Kaggle dataset with `kagglehub`
- organizes files into `data/queries`, `data/cases`, `data/statutes`
- parses raw files into CSV and JSON
- preprocesses legal text
- runs case retrieval with TF-IDF, BM25, and sentence embeddings
- runs statute retrieval with TF-IDF, sentence embeddings, and a multi-label classifier when qrels are available
- evaluates retrieval with Precision@K, Recall@K, F1@K, MAP, and nDCG
- saves outputs under `outputs/`
- serves a local Streamlit dashboard

## Setup

```bash
pip install -r requirements.txt
```

If Kaggle asks for credentials, configure your Kaggle API token before running the pipeline.

## Run The Full Pipeline

```bash
python main.py --top-k 10
```

Useful options:

```bash
python main.py --top-k 20 --force
python main.py --skip-download
python main.py --remove-stopwords
```

Use `--skip-download` after manually placing the dataset under `data/raw`.

By default the supervised statute classifier follows the AILA split: it trains on `AILA_Q1` through `AILA_Q10`, and reported metrics are computed on `AILA_Q11` through `AILA_Q50`. Retrieval result files are still produced for all queries.

## Run The Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard includes model comparison charts, best-model insights, metric tables, and an interactive query explorer for cases and statutes.

## Run Extra GPU Experiments

If PyTorch is CPU-only but CUDA is installed, install a CUDA wheel first. On this machine the working command was:

```bash
pip install --user --upgrade --no-cache-dir torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Then run the fast sweep:

```bash
python experiments.py --mode fast --top-k 10
```

Run the stronger GPU sweep:

```bash
python experiments.py --mode strong --top-k 10
```

Tune lexical, embedding, and classifier fusion weights:

```bash
python tune_fusion.py
```

Experiment outputs are saved under `outputs/experiments/`. The best tuned runs are:

- `outputs/experiments/case_best_tuned.csv`
- `outputs/experiments/case_best_tuned.trec`
- `outputs/experiments/statute_best_tuned.csv`
- `outputs/experiments/statute_best_tuned.trec`

## Project Structure

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
data/
  queries/
  cases/
  statutes/
  qrels/
  processed/
models/
outputs/
```

## Outputs

- `outputs/*_results.csv`: ranked retrieval results
- `outputs/*.trec`: TREC-style run files
- `outputs/*_per_query_metrics.csv`: query-level evaluation
- `outputs/metrics_summary.csv`: model-level comparison
- `outputs/metrics_summary.json`: JSON version of summary metrics
- `models/`: cached vectorizers, BM25 models, classifier, and embeddings

## Notes

The first sentence-transformers run downloads `sentence-transformers/all-MiniLM-L6-v2` and caches corpus embeddings in `models/`. Later runs reuse those embeddings unless `--force` is supplied.
