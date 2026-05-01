import math

import numpy as np
import pandas as pd


def qrels_to_dict(qrels: pd.DataFrame) -> dict[str, dict[str, int]]:
    if qrels is None or qrels.empty:
        return {}
    result: dict[str, dict[str, int]] = {}
    for _, row in qrels.iterrows():
        query_id = str(row["query_id"])
        doc_id = str(row["doc_id"])
        relevance = int(row["relevance"])
        result.setdefault(query_id, {})[doc_id] = relevance
    return result


def precision_at_k(ranked_docs: list[str], relevant_docs: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_docs = ranked_docs[:k]
    hits = sum(1 for doc_id in top_docs if doc_id in relevant_docs)
    return hits / k


def recall_at_k(ranked_docs: list[str], relevant_docs: set[str], k: int) -> float:
    if not relevant_docs:
        return 0.0
    top_docs = ranked_docs[:k]
    hits = sum(1 for doc_id in top_docs if doc_id in relevant_docs)
    return hits / len(relevant_docs)


def f1_at_k(ranked_docs: list[str], relevant_docs: set[str], k: int) -> float:
    precision = precision_at_k(ranked_docs, relevant_docs, k)
    recall = recall_at_k(ranked_docs, relevant_docs, k)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def average_precision_at_k(ranked_docs: list[str], relevant_docs: set[str], k: int) -> float:
    if not relevant_docs:
        return 0.0
    hits = 0
    score = 0.0
    for idx, doc_id in enumerate(ranked_docs[:k], start=1):
        if doc_id in relevant_docs:
            hits += 1
            score += hits / idx
    return score / min(len(relevant_docs), k)


def ndcg_at_k(
    ranked_docs: list[str],
    relevance_by_doc: dict[str, int],
    k: int,
) -> float:
    def dcg(relevances: list[int]) -> float:
        return sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevances))

    gains = [int(relevance_by_doc.get(doc_id, 0)) for doc_id in ranked_docs[:k]]
    ideal = sorted([int(rel) for rel in relevance_by_doc.values() if int(rel) > 0], reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(gains) / ideal_dcg


def evaluate_run(run: pd.DataFrame, qrels: pd.DataFrame, k: int = 10) -> tuple[pd.DataFrame, dict]:
    if run.empty or qrels.empty:
        empty = pd.DataFrame(
            columns=[
                "query_id",
                "precision",
                "recall",
                "f1",
                "average_precision",
                "ndcg",
            ]
        )
        return empty, {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "f1_at_k": 0.0,
            "map": 0.0,
            "ndcg_at_k": 0.0,
            "num_queries": 0,
        }

    qrels_map = qrels_to_dict(qrels)
    rows = []
    for query_id, relevance_by_doc in qrels_map.items():
        relevant_docs = {doc_id for doc_id, rel in relevance_by_doc.items() if int(rel) > 0}
        ranked_docs = (
            run[run["query_id"].astype(str) == str(query_id)]
            .sort_values("rank")["doc_id"]
            .astype(str)
            .tolist()
        )
        rows.append(
            {
                "query_id": query_id,
                "precision": precision_at_k(ranked_docs, relevant_docs, k),
                "recall": recall_at_k(ranked_docs, relevant_docs, k),
                "f1": f1_at_k(ranked_docs, relevant_docs, k),
                "average_precision": average_precision_at_k(ranked_docs, relevant_docs, k),
                "ndcg": ndcg_at_k(ranked_docs, relevance_by_doc, k),
            }
        )

    details = pd.DataFrame(rows)
    summary = {
        "precision_at_k": float(np.mean(details["precision"])) if not details.empty else 0.0,
        "recall_at_k": float(np.mean(details["recall"])) if not details.empty else 0.0,
        "f1_at_k": float(np.mean(details["f1"])) if not details.empty else 0.0,
        "map": float(np.mean(details["average_precision"])) if not details.empty else 0.0,
        "ndcg_at_k": float(np.mean(details["ndcg"])) if not details.empty else 0.0,
        "num_queries": int(len(details)),
    }
    return details, summary
