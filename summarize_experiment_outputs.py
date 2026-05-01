from pathlib import Path

import pandas as pd


OUT_DIR = Path("outputs") / "experiments"
BASELINE = {
    "case_map": 0.1502281746031746,
    "case_ndcg": 0.19210580675104127,
    "statute_map": 0.19247123015873016,
    "statute_ndcg": 0.29985679607882415,
}


def append_row(rows, experiment, change, hypothesis, case_map, case_ndcg, statute_map=None, statute_ndcg=None, notes=""):
    rows.append(
        {
            "experiment": experiment,
            "change_made": change,
            "hypothesis": hypothesis,
            "case_map_before": BASELINE["case_map"],
            "case_map_after": case_map,
            "statute_map_before": BASELINE["statute_map"],
            "statute_map_after": BASELINE["statute_map"] if statute_map is None else statute_map,
            "case_ndcg10_before": BASELINE["case_ndcg"],
            "case_ndcg10_after": case_ndcg,
            "statute_ndcg10_before": BASELINE["statute_ndcg"],
            "statute_ndcg10_after": BASELINE["statute_ndcg"] if statute_ndcg is None else statute_ndcg,
            "notes": notes,
        }
    )


def main():
    log_path = OUT_DIR / "legal_ir_experiment_log.csv"
    rows = pd.read_csv(log_path).to_dict(orient="records")

    query_board = pd.read_csv(OUT_DIR / "query_variant_leaderboard.csv")
    non_control = query_board[query_board["query_col"] != "clean_text"]
    best_case_query = non_control[non_control["task"] == "case"].sort_values("map", ascending=False).iloc[0]
    best_statute_query = non_control[non_control["task"] == "statute"].sort_values("map", ascending=False).iloc[0]
    append_row(
        rows,
        "Query-side truncation/cue filtering",
        "Tested first/last token windows and a legal-cue window for TF-IDF and BM25.",
        "Long situation queries may contain procedural noise; shorter or cue-focused queries might reduce drift.",
        float(best_case_query["map"]),
        float(best_case_query["ndcg_at_k"]),
        float(best_statute_query["map"]),
        float(best_statute_query["ndcg_at_k"]),
        f"Best non-control case variant: {best_case_query['model']}; best non-control statute variant: {best_statute_query['model']}. Full clean_text remained strongest.",
    )

    supervised_rows = []
    for model in ["case_pair_logreg", "case_pair_sgd_log"]:
        path = OUT_DIR / f"{model}.csv"
        if path.exists():
            supervised_rows.append(model)
    if supervised_rows:
        from data_loader import AilaDataLoader
        from evaluation.metrics import evaluate_run
        from main import filter_qrels_to_queries, filter_queries

        bundle = AilaDataLoader(".").load_processed()
        eval_queries = filter_queries(bundle.queries, min_id=11)
        qrels = filter_qrels_to_queries(bundle.case_qrels, eval_queries)
        metrics = []
        for model in supervised_rows:
            _, summary = evaluate_run(pd.read_csv(OUT_DIR / f"{model}.csv"), qrels, k=10)
            metrics.append({**summary, "model": model})
        best_supervised = pd.DataFrame(metrics).sort_values("map", ascending=False).iloc[0]
        pd.DataFrame(metrics).to_csv(OUT_DIR / "case_supervised_partial_metrics.csv", index=False)
        append_row(
            rows,
            "Supervised case pair ranker",
            "Trained local pairwise classifiers on Q1-Q10 case qrels using TF-IDF cosine, term-overlap, length, and BM25 features.",
            "A small supervised signal might learn relevance patterns that pure lexical ranking misses.",
            float(best_supervised["map"]),
            float(best_supervised["ndcg_at_k"]),
            notes=f"Best completed model: {best_supervised['model']}. Tree models were changed to n_jobs=1 after sandboxed thread creation failed.",
        )

    pd.DataFrame(rows).to_csv(log_path, index=False)

    report_path = OUT_DIR / "legal_ir_experiment_report.md"
    report = report_path.read_text(encoding="utf-8")
    extra = [
        "### Experiment: Query-side truncation/cue filtering",
        f"- Change made: Tested first/last token windows and a legal-cue window for TF-IDF and BM25.",
        "- Hypothesis: Long situation queries may contain procedural noise; shorter or cue-focused queries might reduce drift.",
        f"- Case MAP: {BASELINE['case_map']:.4f} -> {float(best_case_query['map']):.4f}",
        f"- Statute MAP: {BASELINE['statute_map']:.4f} -> {float(best_statute_query['map']):.4f}",
        f"- Case nDCG@10: {BASELINE['case_ndcg']:.4f} -> {float(best_case_query['ndcg_at_k']):.4f}",
        f"- Statute nDCG@10: {BASELINE['statute_ndcg']:.4f} -> {float(best_statute_query['ndcg_at_k']):.4f}",
        f"- Notes: Best non-control case variant was {best_case_query['model']}; full clean_text remained stronger. This argues against naive truncation.",
        "",
    ]
    if supervised_rows:
        extra.extend(
            [
                "### Experiment: Supervised case pair ranker",
                "- Change made: Trained local pairwise classifiers on Q1-Q10 case qrels using TF-IDF cosine, term-overlap, length, and BM25 features.",
                "- Hypothesis: A small supervised signal might learn relevance patterns that pure lexical ranking misses.",
                f"- Case MAP: {BASELINE['case_map']:.4f} -> {float(best_supervised['map']):.4f}",
                f"- Statute MAP: {BASELINE['statute_map']:.4f} -> {BASELINE['statute_map']:.4f}",
                f"- Case nDCG@10: {BASELINE['case_ndcg']:.4f} -> {float(best_supervised['ndcg_at_k']):.4f}",
                f"- Statute nDCG@10: {BASELINE['statute_ndcg']:.4f} -> {BASELINE['statute_ndcg']:.4f}",
                f"- Notes: Best completed model was {best_supervised['model']}; Q1-Q10 is too little training signal for this feature-only ranker.",
                "",
            ]
        )
    extra.extend(
        [
            "### Experiment: Enhanced embedding fusion attempt",
            "- Change made: Tried to rebuild candidate pools with cached MiniLM/MPNet plus tuned BM25 and passage BM25.",
            "- Hypothesis: Adding tuned lexical and passage candidates to the existing embedding fusion might improve case recall.",
            f"- Case MAP: {BASELINE['case_map']:.4f} -> not completed",
            f"- Statute MAP: {BASELINE['statute_map']:.4f} -> not run",
            f"- Case nDCG@10: {BASELINE['case_ndcg']:.4f} -> not completed",
            f"- Statute nDCG@10: {BASELINE['statute_ndcg']:.4f} -> not run",
            "- Notes: The run attempted HuggingFace model loading, hit network/proxy failures under sandboxing, then exceeded the escalated timeout. The script remains as a reproducible next step when models are cached locally.",
            "",
        ]
    )
    if "### Experiment: Query-side truncation/cue filtering" not in report:
        report_path.write_text(report.rstrip() + "\n\n" + "\n".join(extra), encoding="utf-8")


if __name__ == "__main__":
    main()
