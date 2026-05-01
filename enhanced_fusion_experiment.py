from itertools import product
from pathlib import Path

import pandas as pd

from data_loader import AilaDataLoader
from evaluation.metrics import evaluate_run
from experiments import fuse_runs
from main import add_clean_columns, filter_qrels_to_queries, filter_queries
from retrieval import BM25Retriever, EmbeddingRetriever, PassageBM25Retriever, TfidfRetriever


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs" / "experiments"


def query_number(query_id: str) -> int:
    return int("".join(ch for ch in str(query_id) if ch.isdigit()))


def load_bundle():
    bundle = AilaDataLoader(ROOT).load_processed()
    if "clean_text" not in bundle.queries.columns:
        bundle = add_clean_columns(bundle)
    for frame in [bundle.queries, bundle.cases, bundle.statutes]:
        frame["clean_text"] = frame["clean_text"].fillna("").astype(str)
    return bundle


def save_run(run: pd.DataFrame, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run.to_csv(OUT_DIR / f"{name}.csv", index=False)
    with (OUT_DIR / f"{name}.trec").open("w", encoding="utf-8") as handle:
        for row in run.sort_values(["query_id", "rank"]).itertuples(index=False):
            handle.write(f"{row.query_id} Q0 {row.doc_id} {int(row.rank)} {float(row.score):.8f} {row.model}\n")


def rrf_fuse(runs: dict[str, pd.DataFrame], model_name: str, task: str, top_k: int, k_const: int = 60):
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


def compositions(names, step=0.1):
    slots = int(round(1 / step))
    for raw in product(range(slots + 1), repeat=len(names)):
        if sum(raw) != slots:
            continue
        weights = dict(zip(names, [value / slots for value in raw]))
        if sum(value > 0 for value in weights.values()) < 2:
            continue
        yield weights


def restrict_run(run, min_id, max_id=None):
    mask = run["query_id"].map(query_number) >= min_id
    if max_id is not None:
        mask &= run["query_id"].map(query_number) <= max_id
    return run[mask].copy()


def grid_search(runs, qrels, task, names, label, tune_min=None, tune_max=None):
    eval_qrels = qrels
    if tune_min is not None:
        eval_qrels = qrels[qrels["query_id"].map(query_number).between(tune_min, tune_max)]
    rows = []
    best = None
    best_run = None
    best_weights = None
    for weights in compositions(names):
        run = fuse_runs({name: runs[name] for name in names}, weights, f"{task}_{label}", task, 10)
        eval_run = restrict_run(run, tune_min, tune_max) if tune_min is not None else run
        _, summary = evaluate_run(eval_run, eval_qrels, k=10)
        row = {**summary, "weights": weights}
        rows.append(row)
        if best is None or summary["map"] > best["map"]:
            best = summary
            best_run = run
            best_weights = weights
    return pd.DataFrame(rows), best_run, best, best_weights


def main():
    bundle = load_bundle()
    eval_queries = filter_queries(bundle.queries, min_id=11)
    qrels = filter_qrels_to_queries(bundle.case_qrels, eval_queries)
    candidate_k = 200

    runs = {
        "tfidf": TfidfRetriever("case_enhanced_tfidf", model_dir=MODEL_DIR).fit(bundle.cases).retrieve_many(bundle.queries, candidate_k, task="case"),
        "bm25": BM25Retriever("case_enhanced_bm25", model_dir=MODEL_DIR).fit(bundle.cases).retrieve_many(bundle.queries, candidate_k, task="case"),
        "bm25_len": BM25Retriever("case_enhanced_bm25_len", model_dir=MODEL_DIR, k1=1.5, b=1.0).fit(bundle.cases).retrieve_many(bundle.queries, candidate_k, task="case"),
        "passage": PassageBM25Retriever("case_enhanced_passage", model_dir=MODEL_DIR, k1=1.5, b=1.0, chunk_size=180, overlap=45, aggregate="sum").fit(bundle.cases).retrieve_many(bundle.queries, candidate_k, task="case"),
        "mini": EmbeddingRetriever(
            "case_exp_all_MiniLM_L6_v2",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_dir=MODEL_DIR,
            batch_size=16,
            max_chars=7000,
        ).fit(bundle.cases).retrieve_many(bundle.queries, candidate_k, task="case"),
        "mpnet": EmbeddingRetriever(
            "case_exp_all_mpnet_base_v2",
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_dir=MODEL_DIR,
            batch_size=16,
            max_chars=7000,
        ).fit(bundle.cases).retrieve_many(bundle.queries, candidate_k, task="case"),
    }
    names = ["tfidf", "bm25", "bm25_len", "passage", "mini", "mpnet"]
    board, best_run, best, weights = grid_search(runs, qrels, "case", names, "enhanced_full")
    board.sort_values("map", ascending=False).to_csv(OUT_DIR / "case_enhanced_fusion_grid.csv", index=False)
    best_run["model"] = "case_enhanced_fusion_full"
    save_run(best_run, "case_enhanced_fusion_full")

    rrf = rrf_fuse(runs, "case_enhanced_rrf", "case", 10)
    save_run(rrf, "case_enhanced_rrf")
    _, rrf_summary = evaluate_run(rrf, qrels, k=10)

    split_board, split_run, split_val, split_weights = grid_search(runs, qrels, "case", names, "enhanced_split", 11, 30)
    split_board.sort_values("map", ascending=False).to_csv(OUT_DIR / "case_enhanced_split_grid.csv", index=False)
    split_run["model"] = "case_enhanced_fusion_split"
    save_run(split_run, "case_enhanced_fusion_split")
    test_qrels = qrels[qrels["query_id"].map(query_number).between(31, 50)]
    _, split_test = evaluate_run(restrict_run(split_run, 31, 50), test_qrels, k=10)

    rows = [
        {"model": "case_enhanced_fusion_full", **best, "weights": weights},
        {"model": "case_enhanced_rrf", **rrf_summary, "weights": "uniform_rrf"},
        {"model": "case_enhanced_fusion_split_val", **split_val, "weights": split_weights},
        {"model": "case_enhanced_fusion_split_test", **split_test, "weights": split_weights},
    ]
    pd.DataFrame(rows).to_csv(OUT_DIR / "case_enhanced_fusion_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
