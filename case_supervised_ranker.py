from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import AilaDataLoader
from evaluation.metrics import evaluate_run
from main import add_clean_columns, filter_qrels_to_queries, filter_queries
from preprocessing import tokenize
from retrieval import BM25Retriever


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs" / "experiments"


def load_bundle():
    bundle = AilaDataLoader(ROOT).load_processed()
    if "clean_text" not in bundle.queries.columns:
        bundle = add_clean_columns(bundle)
    for frame in [bundle.queries, bundle.cases]:
        frame["clean_text"] = frame["clean_text"].fillna("").astype(str)
    return bundle


def build_training_pairs(bundle, neg_multiplier=30):
    qrels = bundle.case_qrels.copy()
    train_queries = set(filter_queries(bundle.queries, max_id=10)["query_id"].astype(str))
    train_qrels = qrels[qrels["query_id"].astype(str).isin(train_queries)]
    positives = train_qrels[train_qrels["relevance"].astype(int) > 0][["query_id", "doc_id"]].copy()
    rng = np.random.default_rng(42)
    rows = []
    for query_id, group in train_qrels.groupby("query_id"):
        pos_docs = set(group[group["relevance"].astype(int) > 0]["doc_id"].astype(str))
        neg_docs = group[group["relevance"].astype(int) <= 0]["doc_id"].astype(str).to_numpy()
        sample_size = min(len(neg_docs), max(200, neg_multiplier * max(1, len(pos_docs))))
        sampled = rng.choice(neg_docs, size=sample_size, replace=False)
        for doc_id in pos_docs:
            rows.append((str(query_id), str(doc_id), 1))
        for doc_id in sampled:
            rows.append((str(query_id), str(doc_id), 0))
    return pd.DataFrame(rows, columns=["query_id", "doc_id", "label"]), positives


def pair_features(bundle, pairs, vectorizer=None, bm25_scores=None):
    queries = bundle.queries.set_index("query_id")["clean_text"].astype(str)
    cases = bundle.cases.set_index("doc_id")["clean_text"].astype(str)
    query_texts = pairs["query_id"].map(queries).fillna("").tolist()
    doc_texts = pairs["doc_id"].map(cases).fillna("").tolist()
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=60000, ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_df=0.95)
        vectorizer.fit(bundle.queries["clean_text"].tolist() + bundle.cases["clean_text"].tolist())

    q_mat = vectorizer.transform(query_texts)
    d_mat = vectorizer.transform(doc_texts)
    cosine = np.asarray(q_mat.multiply(d_mat).sum(axis=1)).ravel()
    q_terms = np.asarray(q_mat.getnnz(axis=1), dtype=float)
    d_terms = np.asarray(d_mat.getnnz(axis=1), dtype=float)
    overlap = []
    containment = []
    for query_text, doc_text in zip(query_texts, doc_texts):
        q_set = set(query_text.split())
        d_set = set(doc_text.split())
        common = q_set & d_set
        overlap.append(len(common) / max(1, len(q_set | d_set)))
        containment.append(len(common) / max(1, len(q_set)))
    features = pd.DataFrame(
        {
            "query_id": pairs["query_id"].astype(str),
            "doc_id": pairs["doc_id"].astype(str),
            "tfidf_cosine": cosine,
            "query_terms": q_terms,
            "doc_terms": d_terms,
            "jaccard": overlap,
            "query_containment": containment,
        }
    )
    if bm25_scores is not None:
        keys = list(zip(features["query_id"], features["doc_id"]))
        features["bm25_score"] = [bm25_scores.get(key, 0.0) for key in keys]
    return features, pairs["label"].astype(int), vectorizer


def build_bm25_score_map(bundle):
    bm25 = BM25Retriever("case_supervised_feature_bm25", model_dir=MODEL_DIR, k1=1.5, b=1.0).fit(bundle.cases)
    doc_ids = bm25.doc_ids.astype(str)
    scores = {}
    for row in bundle.queries.itertuples(index=False):
        query_id = str(row.query_id)
        query_scores = bm25.model.get_scores(tokenize(row.clean_text))
        for doc_id, score in zip(doc_ids, query_scores):
            scores[(query_id, str(doc_id))] = float(score)
    return scores


def rank_all(bundle, model, vectorizer, bm25_scores, name, top_k=10):
    query_ids = bundle.queries["query_id"].astype(str).tolist()
    doc_ids = bundle.cases["doc_id"].astype(str).tolist()
    all_pairs = pd.DataFrame(
        [(query_id, doc_id, 0) for query_id in query_ids for doc_id in doc_ids],
        columns=["query_id", "doc_id", "label"],
    )
    features, _, _ = pair_features(bundle, all_pairs, vectorizer=vectorizer, bm25_scores=bm25_scores)
    x_all = features.drop(columns=["query_id", "doc_id"]).to_numpy()
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x_all)[:, 1]
    else:
        scores = model.decision_function(x_all)
    pred = features[["query_id", "doc_id"]].copy()
    pred["score"] = scores
    ranked = []
    for query_id, group in pred.groupby("query_id"):
        group = group.sort_values("score", ascending=False).head(top_k).copy()
        group["rank"] = range(1, len(group) + 1)
        group["task"] = "case"
        group["model"] = name
        ranked.append(group[["query_id", "task", "model", "doc_id", "rank", "score"]])
    return pd.concat(ranked, ignore_index=True)


def save_run(run, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run.to_csv(OUT_DIR / f"{name}.csv", index=False)
    with (OUT_DIR / f"{name}.trec").open("w", encoding="utf-8") as handle:
        for row in run.sort_values(["query_id", "rank"]).itertuples(index=False):
            handle.write(f"{row.query_id} Q0 {row.doc_id} {int(row.rank)} {float(row.score):.8f} {row.model}\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    train_pairs, positives = build_training_pairs(bundle)
    bm25_scores = build_bm25_score_map(bundle)
    train_features, labels, vectorizer = pair_features(bundle, train_pairs, bm25_scores=bm25_scores)
    x_train = train_features.drop(columns=["query_id", "doc_id"]).to_numpy()
    y_train = labels.to_numpy()

    models = {
        "case_pair_logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "case_pair_sgd_log": make_pipeline(StandardScaler(), SGDClassifier(loss="log_loss", alpha=0.0001, class_weight="balanced", random_state=42, max_iter=3000)),
        "case_pair_rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=1),
        "case_pair_extratrees": ExtraTreesClassifier(n_estimators=500, min_samples_leaf=1, class_weight="balanced", random_state=42, n_jobs=1),
    }
    eval_queries = filter_queries(bundle.queries, min_id=11)
    qrels = filter_qrels_to_queries(bundle.case_qrels, eval_queries)
    rows = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        joblib.dump({"model": model, "vectorizer": vectorizer}, MODEL_DIR / f"{name}.joblib")
        run = rank_all(bundle, model, vectorizer, bm25_scores, name)
        save_run(run, name)
        _, summary = evaluate_run(run, qrels, k=10)
        rows.append({**summary, "task": "case", "model": name, "train_pairs": len(train_pairs), "train_positives": int(labels.sum())})
    board = pd.DataFrame(rows).sort_values("map", ascending=False)
    board.to_csv(OUT_DIR / "case_supervised_ranker_leaderboard.csv", index=False)
    print(board.to_string(index=False))


if __name__ == "__main__":
    main()
