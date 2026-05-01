import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import AilaDataLoader
from evaluation.metrics import evaluate_run
from main import (
    add_clean_columns,
    filter_qrels_to_queries,
    filter_queries,
    query_number,
)
from preprocessing import clean_text
from retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    EmbeddingRetriever,
    StatuteClassifierRetriever,
    TfidfRetriever,
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs" / "experiments"


def minmax(values: pd.Series) -> pd.Series:
    low = values.min()
    high = values.max()
    if high == low:
        return pd.Series(np.ones(len(values)), index=values.index)
    return (values - low) / (high - low)


def normalize_run(run: pd.DataFrame, score_name: str) -> pd.DataFrame:
    frame = run[["query_id", "doc_id", "score"]].copy()
    frame[score_name] = frame.groupby("query_id")["score"].transform(minmax)
    return frame[["query_id", "doc_id", score_name]]


def fuse_runs(
    runs: dict[str, pd.DataFrame],
    weights: dict[str, float],
    model_name: str,
    task: str,
    top_k: int,
) -> pd.DataFrame:
    merged = None
    for name, run in runs.items():
        current = normalize_run(run, name)
        merged = current if merged is None else merged.merge(current, on=["query_id", "doc_id"], how="outer")

    merged = merged.fillna(0.0)
    merged["score"] = 0.0
    for name, weight in weights.items():
        merged["score"] += weight * merged[name]

    ranked = []
    for query_id, group in merged.groupby("query_id"):
        group = group.sort_values("score", ascending=False).head(top_k).copy()
        group["rank"] = range(1, len(group) + 1)
        group["model"] = model_name
        group["task"] = task
        ranked.append(group[["query_id", "task", "model", "doc_id", "rank", "score"]])
    return pd.concat(ranked, ignore_index=True)


def save_run(run: pd.DataFrame, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run.to_csv(OUT_DIR / f"{name}.csv", index=False)
    with (OUT_DIR / f"{name}.trec").open("w", encoding="utf-8") as handle:
        for row in run.sort_values(["query_id", "rank"]).itertuples(index=False):
            handle.write(f"{row.query_id} Q0 {row.doc_id} {int(row.rank)} {float(row.score):.8f} {row.model}\n")


def score_run(
    run: pd.DataFrame,
    qrels: pd.DataFrame,
    task: str,
    model: str,
    k: int,
) -> dict:
    details, summary = evaluate_run(run, qrels, k=k)
    summary.update({"task": task, "model": model, "k": k})
    details.insert(0, "task", task)
    details.insert(1, "model", model)
    details.to_csv(OUT_DIR / f"{task}_{model}_per_query.csv", index=False)
    return summary


def load_bundle():
    loader = AilaDataLoader(ROOT)
    bundle = loader.load_processed()
    if "clean_text" not in bundle.queries.columns:
        bundle = add_clean_columns(bundle)
    else:
        for frame in [bundle.queries, bundle.cases, bundle.statutes]:
            frame["clean_text"] = frame["clean_text"].fillna("").astype(str)
    return bundle


def build_candidate_runs(bundle, task: str, candidate_k: int, embedding_models: list[str], force: bool):
    docs = bundle.cases if task == "case" else bundle.statutes
    queries = bundle.queries
    runs = {}

    tfidf = TfidfRetriever(f"{task}_exp_tfidf", model_dir=MODEL_DIR).fit(docs)
    runs["tfidf"] = tfidf.retrieve_many(queries, top_k=candidate_k, task=task)

    bm25 = BM25Retriever(f"{task}_exp_bm25", model_dir=MODEL_DIR).fit(docs)
    runs["bm25"] = bm25.retrieve_many(queries, top_k=candidate_k, task=task)

    for model_name in embedding_models:
        short = model_name.split("/")[-1].replace("-", "_").replace(".", "_")
        retriever = EmbeddingRetriever(
            f"{task}_exp_{short}",
            model_name=model_name,
            model_dir=MODEL_DIR,
            batch_size=16,
            max_chars=12000,
            window_tokens=512,
            window_stride=256,
            preprocessing_signature="clean_text_v1",
        ).fit(docs, force=force)
        runs[f"emb_{short}"] = retriever.retrieve_many(queries, top_k=candidate_k, task=task)

    if task == "statute":
        train_queries = filter_queries(bundle.queries, max_id=10)
        train_qrels = filter_qrels_to_queries(bundle.statute_qrels, train_queries)
        classifier = StatuteClassifierRetriever("statute_exp_classifier", model_dir=MODEL_DIR).fit(
            train_queries, train_qrels
        )
        runs["classifier"] = classifier.retrieve_many(queries, top_k=candidate_k, task=task)

    return runs


def run_fusion_grid(runs: dict[str, pd.DataFrame], task: str, top_k: int) -> dict[str, pd.DataFrame]:
    grid = {}
    if "tfidf" in runs and "bm25" in runs:
        for w in [0.25, 0.5, 0.75]:
            name = f"{task}_fusion_tfidf{int(w*100)}_bm25{int((1-w)*100)}"
            grid[name] = fuse_runs(
                {"tfidf": runs["tfidf"], "bm25": runs["bm25"]},
                {"tfidf": w, "bm25": 1 - w},
                name,
                task,
                top_k,
            )
    embedding_keys = [key for key in runs if key.startswith("emb_")]
    for emb_key in embedding_keys:
        name = f"{task}_fusion_tfidf_bm25_{emb_key}"
        grid[name] = fuse_runs(
            {"tfidf": runs["tfidf"], "bm25": runs["bm25"], emb_key: runs[emb_key]},
            {"tfidf": 0.45, "bm25": 0.45, emb_key: 0.10},
            name,
            task,
            top_k,
        )
    if task == "statute" and "classifier" in runs:
        name = "statute_fusion_classifier_tfidf_bm25"
        grid[name] = fuse_runs(
            {"classifier": runs["classifier"], "tfidf": runs["tfidf"], "bm25": runs["bm25"]},
            {"classifier": 0.70, "tfidf": 0.20, "bm25": 0.10},
            name,
            task,
            top_k,
        )
    return grid


def rerank_with_cross_encoder(
    bundle,
    candidate_run: pd.DataFrame,
    task: str,
    model_name: str,
    candidate_k: int,
    output_k: int,
    batch_size: int,
    max_doc_chars: int,
) -> pd.DataFrame:
    docs = bundle.cases if task == "case" else bundle.statutes
    short = model_name.split("/")[-1].replace("-", "_")
    reranker = CrossEncoderReranker(
        f"{task}_rerank_{short}",
        model_name=model_name,
        model_dir=MODEL_DIR,
        batch_size=batch_size,
        max_doc_chars=max_doc_chars,
    )
    return reranker.rerank(
        candidate_run,
        bundle.queries,
        docs,
        task=task,
        candidate_k=candidate_k,
        output_k=output_k,
    )


def train_statute_binary_rankers(bundle, top_k: int) -> pd.DataFrame:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    queries = bundle.queries.copy()
    statutes = bundle.statutes.copy()
    qrels = bundle.statute_qrels.copy()
    positives = qrels[qrels["relevance"].astype(int) > 0][["query_id", "doc_id"]].assign(label=1)

    rng = np.random.default_rng(42)
    train_ids = set(filter_queries(queries, max_id=10)["query_id"].astype(str))
    train_rows = []
    pos_set = set(map(tuple, positives[["query_id", "doc_id"]].astype(str).to_numpy()))
    statute_ids = statutes["doc_id"].astype(str).to_numpy()
    for query_id in train_ids:
        pos_docs = [doc_id for qid, doc_id in pos_set if qid == query_id]
        neg_pool = [doc_id for doc_id in statute_ids if (query_id, doc_id) not in pos_set]
        neg_docs = rng.choice(neg_pool, size=min(len(neg_pool), max(30, 8 * len(pos_docs))), replace=False)
        for doc_id in pos_docs:
            train_rows.append((query_id, doc_id, 1))
        for doc_id in neg_docs:
            train_rows.append((query_id, doc_id, 0))

    feature_frame, labels = statute_pair_features(bundle, pd.DataFrame(train_rows, columns=["query_id", "doc_id", "label"]))
    x_train = feature_frame.drop(columns=["query_id", "doc_id"]).to_numpy()
    y_train = labels.to_numpy()

    models = {
        "statute_binary_logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "statute_binary_rf": RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
        "statute_binary_extratrees": ExtraTreesClassifier(n_estimators=350, min_samples_leaf=1, class_weight="balanced", random_state=42, n_jobs=-1),
    }

    all_pairs = pd.DataFrame(
        [(qid, sid, 0) for qid in queries["query_id"].astype(str) for sid in statute_ids],
        columns=["query_id", "doc_id", "label"],
    )
    all_features, _ = statute_pair_features(bundle, all_pairs)
    x_all = all_features.drop(columns=["query_id", "doc_id"]).to_numpy()

    runs = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(x_all)[:, 1]
        else:
            scores = model.decision_function(x_all)
        pred = all_features[["query_id", "doc_id"]].copy()
        pred["score"] = scores
        ranked = []
        for query_id, group in pred.groupby("query_id"):
            group = group.sort_values("score", ascending=False).head(top_k).copy()
            group["rank"] = range(1, len(group) + 1)
            group["task"] = "statute"
            group["model"] = name
            ranked.append(group[["query_id", "task", "model", "doc_id", "rank", "score"]])
        run = pd.concat(ranked, ignore_index=True)
        runs.append((name, run))
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    return dict(runs)


def statute_pair_features(bundle, pairs: pd.DataFrame):
    queries = bundle.queries.set_index("query_id")["clean_text"].astype(str)
    statutes = bundle.statutes.set_index("doc_id")["clean_text"].astype(str)
    query_texts = pairs["query_id"].map(queries).fillna("").tolist()
    doc_texts = pairs["doc_id"].map(statutes).fillna("").tolist()

    vectorizer_path = MODEL_DIR / "statute_pair_feature_tfidf.joblib"
    if vectorizer_path.exists():
        vectorizer = joblib.load(vectorizer_path)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True)
        vectorizer.fit(bundle.queries["clean_text"].tolist() + bundle.statutes["clean_text"].tolist())
        joblib.dump(vectorizer, vectorizer_path)

    q_mat = vectorizer.transform(query_texts)
    d_mat = vectorizer.transform(doc_texts)
    cosine = np.asarray(q_mat.multiply(d_mat).sum(axis=1)).ravel()
    q_len = np.asarray(q_mat.getnnz(axis=1), dtype=float)
    d_len = np.asarray(d_mat.getnnz(axis=1), dtype=float)
    overlap = []
    for query_text, doc_text in zip(query_texts, doc_texts):
        q_terms = set(query_text.split())
        d_terms = set(doc_text.split())
        overlap.append(len(q_terms & d_terms) / max(1, len(q_terms)))
    features = pd.DataFrame(
        {
            "query_id": pairs["query_id"].astype(str),
            "doc_id": pairs["doc_id"].astype(str),
            "cosine": cosine,
            "query_terms": q_len,
            "doc_terms": d_len,
            "term_overlap": overlap,
        }
    )
    return features, pairs["label"].astype(int)


def experiment(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()

    eval_queries = filter_queries(bundle.queries, min_id=args.eval_query_min)
    case_qrels = filter_qrels_to_queries(bundle.case_qrels, eval_queries)
    statute_qrels = filter_qrels_to_queries(bundle.statute_qrels, eval_queries)

    if args.mode == "fast":
        embedding_models = ["sentence-transformers/all-mpnet-base-v2"]
        cross_models = [args.rerank_model]
        candidate_k = args.candidate_k or 60
        rerank_k = args.rerank_k or 50
        batch_size = 24
        max_doc_chars = 3000
    else:
        embedding_models = ["sentence-transformers/all-mpnet-base-v2"]
        cross_models = [args.rerank_model]
        candidate_k = args.candidate_k or 200
        rerank_k = args.rerank_k or 50
        batch_size = 12
        max_doc_chars = 7000

    leaderboard = []
    best_by_task = {}

    for task, qrels in [("case", case_qrels), ("statute", statute_qrels)]:
        runs = build_candidate_runs(bundle, task, candidate_k, embedding_models, args.force_embeddings)
        run_bank = {}
        for name, run in runs.items():
            model_name = f"{task}_{name}"
            final = run.sort_values(["query_id", "rank"]).groupby("query_id").head(args.top_k).copy()
            final["model"] = model_name
            save_run(final, model_name)
            summary = score_run(final, qrels, task, model_name, args.top_k)
            leaderboard.append(summary)
            run_bank[model_name] = final

        fusions = run_fusion_grid(runs, task, args.top_k)
        wide_fusions = run_fusion_grid(runs, task, candidate_k)
        for name, final in fusions.items():
            save_run(final, name)
            summary = score_run(final, qrels, task, name, args.top_k)
            leaderboard.append(summary)
            run_bank[name] = final

        if task == "statute":
            for name, final in train_statute_binary_rankers(bundle, args.top_k).items():
                save_run(final, name)
                summary = score_run(final, qrels, task, name, args.top_k)
                leaderboard.append(summary)
                run_bank[name] = final

        task_scores = pd.DataFrame([row for row in leaderboard if row["task"] == task])
        best_name = task_scores.sort_values("map", ascending=False).iloc[0]["model"]
        best_candidate = (
            wide_fusions.get(best_name)
            if best_name in wide_fusions
            else runs.get(best_name.replace(f"{task}_", ""))
        )
        if best_candidate is None:
            best_candidate = run_bank[best_name]
        best_by_task[task] = str(best_name)

        should_rerank = args.rerank and (args.rerank_task == "all" or args.rerank_task == task)
        if should_rerank:
            rerank_source = runs["bm25"] if task == "case" and "bm25" in runs else best_candidate
            for cross_model in cross_models:
                reranked = rerank_with_cross_encoder(
                    bundle,
                    rerank_source,
                    task,
                    cross_model,
                    candidate_k=rerank_k,
                    output_k=args.top_k,
                    batch_size=batch_size,
                    max_doc_chars=max_doc_chars,
                )
                model_name = reranked["model"].iloc[0]
                save_run(reranked, model_name)
                summary = score_run(reranked, qrels, task, model_name, args.top_k)
                leaderboard.append(summary)

    board = pd.DataFrame(leaderboard).sort_values(["task", "map"], ascending=[True, False])
    board.to_csv(OUT_DIR / f"leaderboard_{args.mode}.csv", index=False)
    (OUT_DIR / f"leaderboard_{args.mode}.json").write_text(
        json.dumps(board.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    print(board.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Additional AILA retrieval experiments")
    parser.add_argument("--mode", choices=["fast", "strong"], default="fast")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--rerank-k", type=int, default=None)
    parser.add_argument("--eval-query-min", type=int, default=11)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--rerank", action="store_true", help="Run cross-encoder reranking.")
    parser.add_argument(
        "--rerank-task",
        choices=["case", "statute", "all"],
        default="case",
        help="Task to rerank when --rerank is set.",
    )
    parser.add_argument(
        "--rerank-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="HuggingFace cross-encoder model for reranking.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    experiment(parse_args())
