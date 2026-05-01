import argparse
import json
from pathlib import Path

import pandas as pd

from data_loader import AilaDataLoader, DatasetBundle
from evaluation.metrics import evaluate_run
from preprocessing import preprocess_corpus
from retrieval import (
    BM25Retriever,
    EmbeddingRetriever,
    StatuteClassifierRetriever,
    TfidfRetriever,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"


def add_clean_columns(bundle: DatasetBundle, remove_stopwords: bool = False) -> DatasetBundle:
    bundle.queries["clean_text"] = preprocess_corpus(
        bundle.queries["text"], remove_stopwords=remove_stopwords
    )
    bundle.cases["clean_text"] = preprocess_corpus(
        bundle.cases["text"], remove_stopwords=remove_stopwords
    )
    bundle.statutes["clean_text"] = preprocess_corpus(
        bundle.statutes["text"], remove_stopwords=remove_stopwords
    )
    return bundle


def save_clean_processed(bundle: DatasetBundle, loader: AilaDataLoader) -> None:
    loader.processed_dir.mkdir(parents=True, exist_ok=True)
    bundle.queries.to_csv(loader.processed_dir / "queries.csv", index=False)
    bundle.cases.to_csv(loader.processed_dir / "cases.csv", index=False)
    bundle.statutes.to_csv(loader.processed_dir / "statutes.csv", index=False)


def query_number(query_id: str) -> int | None:
    digits = "".join(ch for ch in str(query_id) if ch.isdigit())
    return int(digits) if digits else None


def filter_queries(queries: pd.DataFrame, min_id: int | None = None, max_id: int | None = None) -> pd.DataFrame:
    mask = pd.Series(True, index=queries.index)
    numbers = queries["query_id"].map(query_number)
    if min_id is not None:
        mask &= numbers >= min_id
    if max_id is not None:
        mask &= numbers <= max_id
    return queries[mask].copy()


def filter_qrels_to_queries(qrels: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    if qrels.empty:
        return qrels
    query_ids = set(queries["query_id"].astype(str))
    return qrels[qrels["query_id"].astype(str).isin(query_ids)].copy()


def save_run(run: pd.DataFrame, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run.to_csv(OUTPUT_DIR / filename, index=False)
    trec_path = OUTPUT_DIR / filename.replace(".csv", ".trec")
    with trec_path.open("w", encoding="utf-8") as handle:
        for row in run.sort_values(["query_id", "rank"]).itertuples(index=False):
            handle.write(
                f"{row.query_id} Q0 {row.doc_id} {int(row.rank)} {float(row.score):.8f} {row.model}\n"
            )


def evaluate_and_store(run: pd.DataFrame, qrels: pd.DataFrame, task: str, model: str, k: int) -> dict:
    details, summary = evaluate_run(run, qrels, k=k)
    summary.update({"task": task, "model": model, "k": k})
    details.insert(0, "task", task)
    details.insert(1, "model", model)
    details.to_csv(OUTPUT_DIR / f"{task}_{model}_per_query_metrics.csv", index=False)
    return summary


def run_case_retrieval(
    bundle: DatasetBundle,
    eval_qrels: pd.DataFrame,
    top_k: int,
    force: bool,
) -> tuple[list[pd.DataFrame], list[dict]]:
    runs = []
    metrics = []

    tfidf = TfidfRetriever("case_tfidf", model_dir=MODEL_DIR).fit(bundle.cases)
    tfidf.save()
    run = tfidf.retrieve_many(bundle.queries, top_k=top_k, task="case")
    save_run(run, "case_tfidf_results.csv")
    runs.append(run)
    metrics.append(evaluate_and_store(run, eval_qrels, "case", "case_tfidf", top_k))

    bm25 = BM25Retriever("case_bm25", model_dir=MODEL_DIR).fit(bundle.cases)
    bm25.save()
    run = bm25.retrieve_many(bundle.queries, top_k=top_k, task="case")
    save_run(run, "case_bm25_results.csv")
    runs.append(run)
    metrics.append(evaluate_and_store(run, eval_qrels, "case", "case_bm25", top_k))

    embedding = EmbeddingRetriever("case_embeddings", model_dir=MODEL_DIR).fit(
        bundle.cases, force=force
    )
    run = embedding.retrieve_many(bundle.queries, top_k=top_k, task="case")
    save_run(run, "case_embeddings_results.csv")
    runs.append(run)
    metrics.append(evaluate_and_store(run, eval_qrels, "case", "case_embeddings", top_k))

    return runs, metrics


def run_statute_retrieval(
    bundle: DatasetBundle,
    eval_qrels: pd.DataFrame,
    top_k: int,
    force: bool,
    train_query_max: int,
) -> tuple[list[pd.DataFrame], list[dict]]:
    runs = []
    metrics = []

    tfidf = TfidfRetriever("statute_tfidf", model_dir=MODEL_DIR).fit(bundle.statutes)
    tfidf.save()
    run = tfidf.retrieve_many(bundle.queries, top_k=top_k, task="statute")
    save_run(run, "statute_tfidf_results.csv")
    runs.append(run)
    metrics.append(evaluate_and_store(run, eval_qrels, "statute", "statute_tfidf", top_k))

    embedding = EmbeddingRetriever("statute_embeddings", model_dir=MODEL_DIR).fit(
        bundle.statutes, force=force
    )
    run = embedding.retrieve_many(bundle.queries, top_k=top_k, task="statute")
    save_run(run, "statute_embeddings_results.csv")
    runs.append(run)
    metrics.append(
        evaluate_and_store(run, eval_qrels, "statute", "statute_embeddings", top_k)
    )

    try:
        train_queries = filter_queries(bundle.queries, max_id=train_query_max)
        train_qrels = filter_qrels_to_queries(bundle.statute_qrels, train_queries)
        classifier = StatuteClassifierRetriever(model_dir=MODEL_DIR).fit(
            train_queries, train_qrels
        )
        classifier.save()
        run = classifier.retrieve_many(bundle.queries, top_k=top_k, task="statute")
        save_run(run, "statute_classifier_results.csv")
        runs.append(run)
        metrics.append(
            evaluate_and_store(run, eval_qrels, "statute", "statute_classifier", top_k)
        )
    except ValueError as exc:
        print(f"Skipping statute classifier: {exc}")

    return runs, metrics


def run_pipeline(args: argparse.Namespace) -> None:
    loader = AilaDataLoader(ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if args.force or not (loader.processed_dir / "queries.csv").exists():
        source = None if args.skip_download else loader.download_dataset()
        loader.prepare_dataset(source, force=args.force)
        bundle = loader.parse_all()
    else:
        bundle = loader.load_processed()

    bundle = add_clean_columns(bundle, remove_stopwords=args.remove_stopwords)
    save_clean_processed(bundle, loader)
    eval_queries = filter_queries(bundle.queries, min_id=args.eval_query_min)
    case_eval_qrels = filter_qrels_to_queries(bundle.case_qrels, eval_queries)
    statute_eval_qrels = filter_qrels_to_queries(bundle.statute_qrels, eval_queries)

    all_runs = []
    summaries = []
    case_runs, case_metrics = run_case_retrieval(
        bundle, case_eval_qrels, args.top_k, args.force
    )
    statute_runs, statute_metrics = run_statute_retrieval(
        bundle,
        statute_eval_qrels,
        args.top_k,
        args.force,
        args.train_query_max,
    )
    all_runs.extend(case_runs)
    all_runs.extend(statute_runs)
    summaries.extend(case_metrics)
    summaries.extend(statute_metrics)

    combined_runs = pd.concat(all_runs, ignore_index=True) if all_runs else pd.DataFrame()
    combined_runs.to_csv(OUTPUT_DIR / "all_retrieval_results.csv", index=False)

    metrics_df = pd.DataFrame(summaries)
    metrics_df.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    (OUTPUT_DIR / "metrics_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )

    best = {}
    if not metrics_df.empty:
        for task, group in metrics_df.groupby("task"):
            best[task] = group.sort_values("map", ascending=False).iloc[0].to_dict()
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(
            {
                "top_k": args.top_k,
                "remove_stopwords": args.remove_stopwords,
                "classifier_train_query_max": args.train_query_max,
                "eval_query_min": args.eval_query_min,
                "best_models": best,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Pipeline complete. Outputs saved to {OUTPUT_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FIRE 2019 AILA legal retrieval pipeline")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results per query")
    parser.add_argument("--force", action="store_true", help="Rebuild parsed data and cached models")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use files already placed under data/raw instead of downloading from Kaggle",
    )
    parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Remove English stopwords during preprocessing",
    )
    parser.add_argument(
        "--train-query-max",
        type=int,
        default=10,
        help="Highest AILA query number used to train the optional statute classifier",
    )
    parser.add_argument(
        "--eval-query-min",
        type=int,
        default=11,
        help="Lowest AILA query number included in metric evaluation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
