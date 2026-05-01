import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import AilaDataLoader
from evaluation.metrics import evaluate_run
from experiments import fuse_runs, save_run
from main import add_clean_columns, filter_qrels_to_queries, filter_queries
from retrieval import BM25Retriever, EmbeddingRetriever, StatuteClassifierRetriever, TfidfRetriever


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs" / "experiments"
VAL_MIN = 11
VAL_MAX = 30
TEST_MIN = 31
TEST_MAX = 50


def load_bundle():
    bundle = AilaDataLoader(ROOT).load_processed()
    if "clean_text" not in bundle.queries.columns:
        bundle = add_clean_columns(bundle)
    else:
        for frame in [bundle.queries, bundle.cases, bundle.statutes]:
            frame["clean_text"] = frame["clean_text"].fillna("").astype(str)
    return bundle


def query_number(query_id: str) -> int:
    digits = "".join(ch for ch in str(query_id) if ch.isdigit())
    return int(digits) if digits else -1


def restrict_run(run: pd.DataFrame, min_id: int, max_id: int) -> pd.DataFrame:
    numbers = run["query_id"].map(query_number)
    return run[numbers.between(min_id, max_id)].copy()


def restrict_qrels(qrels: pd.DataFrame, min_id: int, max_id: int) -> pd.DataFrame:
    numbers = qrels["query_id"].map(query_number)
    return qrels[numbers.between(min_id, max_id)].copy()


def rrf_fuse(runs, model_name, task, top_k, k_const=60):
    scores = {}
    for run in runs.values():
        for row in run.itertuples(index=False):
            key = (str(row.query_id), str(row.doc_id))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_const + int(row.rank))
    frame = pd.DataFrame(
        [(qid, doc_id, score) for (qid, doc_id), score in scores.items()],
        columns=["query_id", "doc_id", "score"],
    )
    ranked = []
    for query_id, group in frame.groupby("query_id"):
        group = group.sort_values("score", ascending=False).head(top_k).copy()
        group["rank"] = range(1, len(group) + 1)
        group["task"] = task
        group["model"] = model_name
        ranked.append(group[["query_id", "task", "model", "doc_id", "rank", "score"]])
    return pd.concat(ranked, ignore_index=True)


def summarize(run, qrels, task, model):
    _, summary = evaluate_run(run, qrels, k=10)
    summary.update({"task": task, "model": model, "k": 10})
    return summary


def with_split_metadata(
    test_summary: dict,
    val_summary: dict | None,
    split_used: str,
    weights: dict[str, float] | str,
    selected_by: str,
) -> dict:
    row = dict(test_summary)
    row.update(
        {
            "val_map": np.nan if val_summary is None else val_summary["map"],
            "val_ndcg_at_k": np.nan if val_summary is None else val_summary["ndcg_at_k"],
            "weights": json.dumps(weights, sort_keys=True) if isinstance(weights, dict) else weights,
            "split_used": split_used,
            "selected_by": selected_by,
        }
    )
    return row


def grid_search(task, runs, val_qrels, test_qrels, names, step=0.1):
    rows = []
    best_val = None
    best_test = None
    best_run = None
    best_weights = None

    slots = int(round(1 / step))
    for raw in product(range(slots + 1), repeat=len(names)):
        if sum(raw) != slots:
            continue
        weights = dict(zip(names, [value / slots for value in raw]))
        if any(weight > 0.85 for weight in weights.values()) and len(names) > 2:
            continue
        label = "_".join(f"{name}{int(weight*100):02d}" for name, weight in weights.items() if weight > 0)
        model = f"{task}_grid_{label}"
        run = fuse_runs({name: runs[name] for name in names}, weights, model, task, 10)
        val_summary = summarize(restrict_run(run, VAL_MIN, VAL_MAX), val_qrels, task, model)
        test_summary = summarize(restrict_run(run, TEST_MIN, TEST_MAX), test_qrels, task, model)
        rows.append(
            with_split_metadata(
                test_summary,
                val_summary,
                f"val Q{VAL_MIN}-Q{VAL_MAX} / test Q{TEST_MIN}-Q{TEST_MAX}",
                weights,
                "weighted_grid_val_map",
            )
        )
        if best_val is None or (val_summary["map"], val_summary["ndcg_at_k"]) > (
            best_val["map"],
            best_val["ndcg_at_k"],
        ):
            best_val = val_summary
            best_test = test_summary
            best_run = run
            best_weights = weights

    best_test = with_split_metadata(
        best_test,
        best_val,
        f"val Q{VAL_MIN}-Q{VAL_MAX} / test Q{TEST_MIN}-Q{TEST_MAX}",
        best_weights,
        "weighted_grid_val_map",
    )
    return pd.DataFrame(rows), best_test, best_run


def evaluate_rrf(task, runs, val_qrels, test_qrels, names):
    model = f"{task}_rrf_k60"
    run = rrf_fuse({name: runs[name] for name in names}, model, task, 10, k_const=60)
    val_summary = summarize(restrict_run(run, VAL_MIN, VAL_MAX), val_qrels, task, model)
    test_summary = summarize(restrict_run(run, TEST_MIN, TEST_MAX), test_qrels, task, model)
    row = with_split_metadata(
        test_summary,
        val_summary,
        f"RRF no tuning / test Q{TEST_MIN}-Q{TEST_MAX}",
        "uniform_rrf_k60",
        "no_tuning",
    )
    return row, run


def tune_case_bm25(bundle, val_qrels, candidate_k: int):
    rows = []
    best = None
    best_params = (1.5, 0.75)
    for k1 in [0.5, 1.0, 1.5, 2.0]:
        for b in [0.5, 0.75, 1.0]:
            model = f"case_bm25_k1_{str(k1).replace('.', '_')}_b_{str(b).replace('.', '_')}"
            run = (
                BM25Retriever(model, model_dir=MODEL_DIR, k1=k1, b=b)
                .fit(bundle.cases)
                .retrieve_many(bundle.queries, candidate_k, task="case")
            )
            top10 = run.sort_values(["query_id", "rank"]).groupby("query_id").head(10).copy()
            summary = summarize(restrict_run(top10, VAL_MIN, VAL_MAX), val_qrels, "case", model)
            rows.append({**summary, "k1": k1, "b": b})
            if best is None or (summary["map"], summary["ndcg_at_k"]) > (
                best["map"],
                best["ndcg_at_k"],
            ):
                best = summary
                best_params = (k1, b)
    pd.DataFrame(rows).sort_values("map", ascending=False).to_csv(
        OUT_DIR / "case_bm25_validation_grid.csv", index=False
    )
    return best_params, best


def main(args: argparse.Namespace):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    eval_queries = filter_queries(bundle.queries, min_id=VAL_MIN, max_id=TEST_MAX)
    qrels = {
        "case": filter_qrels_to_queries(bundle.case_qrels, eval_queries),
        "statute": filter_qrels_to_queries(bundle.statute_qrels, eval_queries),
    }
    val_qrels = {task: restrict_qrels(task_qrels, VAL_MIN, VAL_MAX) for task, task_qrels in qrels.items()}
    test_qrels = {task: restrict_qrels(task_qrels, TEST_MIN, TEST_MAX) for task, task_qrels in qrels.items()}
    candidate_k = 200
    (case_k1, case_b), bm25_val = tune_case_bm25(bundle, val_qrels["case"], candidate_k)
    print(f"Best case BM25 on validation: k1={case_k1}, b={case_b}, MAP={bm25_val['map']:.4f}")

    case_runs = {
        "tfidf": TfidfRetriever("case_tune_tfidf", model_dir=MODEL_DIR).fit(bundle.cases).retrieve_many(bundle.queries, candidate_k, task="case"),
        "bm25": BM25Retriever("case_tune_bm25", model_dir=MODEL_DIR, k1=case_k1, b=case_b).fit(bundle.cases).retrieve_many(bundle.queries, candidate_k, task="case"),
    }
    if not args.skip_embeddings:
        case_runs["mpnet"] = EmbeddingRetriever(
            "case_exp_all_mpnet_base_v2",
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_dir=MODEL_DIR,
            max_chars=args.embedding_max_chars,
            window_tokens=512 if args.embedding_sliding_windows else None,
            window_stride=256,
            preprocessing_signature="clean_text_v1",
        ).fit(bundle.cases).retrieve_many(bundle.queries, candidate_k, task="case")

    train_queries = filter_queries(bundle.queries, max_id=10)
    train_qrels = filter_qrels_to_queries(bundle.statute_qrels, train_queries)
    statute_runs = {
        "tfidf": TfidfRetriever("statute_tune_tfidf", model_dir=MODEL_DIR).fit(bundle.statutes).retrieve_many(bundle.queries, candidate_k, task="statute"),
        "bm25": BM25Retriever("statute_tune_bm25", model_dir=MODEL_DIR).fit(bundle.statutes).retrieve_many(bundle.queries, candidate_k, task="statute"),
        "classifier": StatuteClassifierRetriever("statute_tune_classifier", model_dir=MODEL_DIR)
        .fit(train_queries, train_qrels)
        .retrieve_many(bundle.queries, candidate_k, task="statute"),
    }
    if not args.skip_embeddings:
        statute_runs["mpnet"] = EmbeddingRetriever(
            "statute_exp_all_mpnet_base_v2",
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_dir=MODEL_DIR,
            max_chars=args.embedding_max_chars,
            window_tokens=512 if args.embedding_sliding_windows else None,
            window_stride=256,
            preprocessing_signature="clean_text_v1",
        ).fit(bundle.statutes).retrieve_many(bundle.queries, candidate_k, task="statute")

    all_rows = []
    case_names = [name for name in ["tfidf", "bm25", "mpnet"] if name in case_runs]
    statute_names = [
        name for name in ["classifier", "tfidf", "bm25", "mpnet"] if name in statute_runs
    ]
    case_grid, case_best, case_best_run = grid_search(
        "case", case_runs, val_qrels["case"], test_qrels["case"], case_names, step=0.1
    )
    statute_grid, statute_best, statute_best_run = grid_search(
        "statute", statute_runs, val_qrels["statute"], test_qrels["statute"], statute_names, step=0.1
    )
    case_rrf, case_rrf_run = evaluate_rrf("case", case_runs, val_qrels["case"], test_qrels["case"], case_names)
    statute_rrf, statute_rrf_run = evaluate_rrf("statute", statute_runs, val_qrels["statute"], test_qrels["statute"], statute_names)
    all_rows.extend(case_grid.to_dict(orient="records"))
    all_rows.extend(statute_grid.to_dict(orient="records"))
    all_rows.extend([case_rrf, statute_rrf])

    if case_rrf["map"] > case_best["map"]:
        case_best = case_rrf
        case_best_run = case_rrf_run
    if statute_rrf["map"] > statute_best["map"]:
        statute_best = statute_rrf
        statute_best_run = statute_rrf_run

    leaderboard = pd.DataFrame(all_rows).sort_values(["task", "map"], ascending=[True, False])
    leaderboard.to_csv(OUT_DIR / "fusion_tuning_leaderboard.csv", index=False)
    (OUT_DIR / "fusion_tuning_leaderboard.json").write_text(
        json.dumps(leaderboard.head(100).to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    save_run(restrict_run(case_best_run, TEST_MIN, TEST_MAX), "case_best_tuned")
    save_run(restrict_run(statute_best_run, TEST_MIN, TEST_MAX), "statute_best_tuned")
    pd.DataFrame([case_best, statute_best]).to_csv(OUT_DIR / "best_tuned_summary.csv", index=False)
    print(pd.DataFrame([case_best, statute_best]).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split-safe AILA fusion tuning")
    parser.add_argument("--embedding-max-chars", type=int, default=7000)
    parser.add_argument(
        "--embedding-sliding-windows",
        action="store_true",
        help="Use mean pooling over 512-token windows with stride 256 for embeddings.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Run split-safe fusion with only locally buildable lexical/classifier signals.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
