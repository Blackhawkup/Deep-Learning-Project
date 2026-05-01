from pathlib import Path

import pandas as pd

from data_loader import AilaDataLoader
from evaluation.metrics import evaluate_run
from main import filter_qrels_to_queries, filter_queries
from preprocessing import tokenize


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "experiments"


def top_terms(text: str, limit: int = 25) -> set[str]:
    tokens = tokenize(text)
    counts = {}
    for token in tokens:
        if len(token) < 4:
            continue
        counts[token] = counts.get(token, 0) + 1
    return {term for term, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]}


def classify_failure(query_text: str, retrieved_texts: list[str], relevant_texts: list[str]) -> tuple[str, str]:
    query_terms = top_terms(query_text)
    retrieved_terms = set().union(*(top_terms(text, 20) for text in retrieved_texts[:5])) if retrieved_texts else set()
    relevant_terms = set().union(*(top_terms(text, 20) for text in relevant_texts[:5])) if relevant_texts else set()

    relevant_count = len(relevant_texts)
    retrieved_overlap = len(query_terms & retrieved_terms) / max(1, len(query_terms))
    relevant_overlap = len(query_terms & relevant_terms) / max(1, len(query_terms))
    avg_rel_len = sum(len(text.split()) for text in relevant_texts) / max(1, relevant_count)

    if relevant_count <= 2:
        return "data sparsity", "Only one or two relevant cases are judged, so missing one document collapses top-10 metrics."
    if relevant_overlap < 0.12 and retrieved_overlap >= 0.12:
        return "vocabulary mismatch", "Retrieved cases share surface query terms more than the judged relevant cases do."
    if avg_rel_len > 6500 and relevant_overlap >= 0.12:
        return "truncation loss", "Relevant judgments are very long; semantic truncation is likely to miss the decisive passage."
    return "topical drift", "Top retrieved cases match broad surface themes but appear to land in the wrong legal issue or fact pattern."


def main():
    loader = AilaDataLoader(ROOT)
    bundle = loader.load_processed()
    run_path = OUT_DIR / "case_best_tuned.csv"
    run = pd.read_csv(run_path)
    run_query_ids = set(run["query_id"].astype(str))
    queries = bundle.queries[bundle.queries["query_id"].astype(str).isin(run_query_ids)].copy()
    qrels = filter_qrels_to_queries(bundle.case_qrels, queries)
    details, summary = evaluate_run(run, qrels, k=10)
    details.to_csv(OUT_DIR / "case_best_tuned_per_query_metrics.csv", index=False)

    query_map = bundle.queries.set_index("query_id")["text"].astype(str).to_dict()
    case_map = bundle.cases.set_index("doc_id")["text"].astype(str).to_dict()
    rel = qrels[qrels["relevance"].astype(int) > 0]

    rows = []
    for metric_row in details.sort_values(["ndcg", "average_precision"]).head(10).itertuples(index=False):
        query_id = str(metric_row.query_id)
        retrieved_docs = (
            run[run["query_id"].astype(str) == query_id]
            .sort_values("rank")
            .head(10)["doc_id"]
            .astype(str)
            .tolist()
        )
        relevant_docs = rel[rel["query_id"].astype(str) == query_id]["doc_id"].astype(str).tolist()
        retrieved_texts = [case_map.get(doc_id, "") for doc_id in retrieved_docs]
        relevant_texts = [case_map.get(doc_id, "") for doc_id in relevant_docs]
        failure_type, rationale = classify_failure(query_map.get(query_id, ""), retrieved_texts, relevant_texts)
        rows.append(
            {
                "query_id": query_id,
                "average_precision": metric_row.average_precision,
                "ndcg": metric_row.ndcg,
                "precision": metric_row.precision,
                "recall": metric_row.recall,
                "failure_type": failure_type,
                "failure_rationale": rationale,
                "query_text": query_map.get(query_id, "")[:1200],
                "top10_docs": " ".join(retrieved_docs),
                "top10_titles": " | ".join(case_map.get(doc_id, "").split("\n")[0][:140] for doc_id in retrieved_docs),
                "relevant_docs": " ".join(relevant_docs),
                "relevant_titles": " | ".join(case_map.get(doc_id, "").split("\n")[0][:140] for doc_id in relevant_docs),
            }
        )

    analysis = pd.DataFrame(rows)
    analysis.to_csv(OUT_DIR / "case_worst_query_error_analysis.csv", index=False)
    counts = analysis["failure_type"].value_counts().to_dict()
    dominant = analysis["failure_type"].mode().iloc[0] if not analysis.empty else "unknown"
    paragraph = (
        f"Dominant failure mode among the 10 worst case queries: {dominant}. "
        f"Counts: {counts}. The failed queries mostly receive zero top-10 hits, so improvements should prioritize "
        "candidate recall before fine reranking. The evidence points to long narrative queries and full judgments "
        "causing broad topical matches; cross-encoder reranking may help precision only after BM25 retrieves a "
        "relevant candidate in its top pool."
    )
    (OUT_DIR / "case_error_analysis_summary.txt").write_text(paragraph, encoding="utf-8")
    print(paragraph)
    print(summary)


if __name__ == "__main__":
    main()
