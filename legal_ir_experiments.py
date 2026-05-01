import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import AilaDataLoader
from evaluation.metrics import evaluate_run
from experiments import fuse_runs
from main import add_clean_columns, filter_qrels_to_queries, filter_queries
from retrieval import BM25Retriever, PassageBM25Retriever, StatuteClassifierRetriever, TfidfRetriever


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs" / "experiments"
BASELINE = {
    "case": {"map": 0.1502281746031746, "ndcg_at_k": 0.19210580675104127},
    "statute": {"map": 0.19247123015873016, "ndcg_at_k": 0.29985679607882415},
}


def query_number(query_id: str) -> int:
    return int("".join(ch for ch in str(query_id) if ch.isdigit()))


def load_bundle():
    bundle = AilaDataLoader(ROOT).load_processed()
    if "clean_text" not in bundle.queries.columns:
        bundle = add_clean_columns(bundle)
    for frame in [bundle.queries, bundle.cases, bundle.statutes]:
        frame["clean_text"] = frame["clean_text"].fillna("").astype(str)
        frame["text"] = frame["text"].fillna("").astype(str)
    return bundle


def save_run(run: pd.DataFrame, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run.to_csv(OUT_DIR / f"{name}.csv", index=False)
    with (OUT_DIR / f"{name}.trec").open("w", encoding="utf-8") as handle:
        for row in run.sort_values(["query_id", "rank"]).itertuples(index=False):
            handle.write(f"{row.query_id} Q0 {row.doc_id} {int(row.rank)} {float(row.score):.8f} {row.model}\n")


def summarize(run: pd.DataFrame, qrels: pd.DataFrame, task: str, model: str, k: int = 10):
    details, summary = evaluate_run(run, qrels, k=k)
    summary.update({"task": task, "model": model, "k": k})
    details.insert(0, "task", task)
    details.insert(1, "model", model)
    details.to_csv(OUT_DIR / f"{task}_{model}_per_query.csv", index=False)
    return details, summary


def normalize_queries_qrels(bundle, min_id: int, max_id: int | None = None):
    queries = filter_queries(bundle.queries, min_id=min_id, max_id=max_id)
    return {
        "queries": queries,
        "case": filter_qrels_to_queries(bundle.case_qrels, queries),
        "statute": filter_qrels_to_queries(bundle.statute_qrels, queries),
    }


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


def restrict_run(run: pd.DataFrame, min_id: int, max_id: int | None = None) -> pd.DataFrame:
    mask = run["query_id"].map(query_number) >= min_id
    if max_id is not None:
        mask &= run["query_id"].map(query_number) <= max_id
    return run[mask].copy()


def composition_weights(names: list[str], step: float = 0.1):
    slots = int(round(1 / step))
    for raw in product(range(slots + 1), repeat=len(names)):
        if sum(raw) != slots:
            continue
        weights = dict(zip(names, [value / slots for value in raw]))
        if sum(weight > 0 for weight in weights.values()) < 2:
            continue
        yield weights


def tune_weighted_fusion(
    runs: dict[str, pd.DataFrame],
    qrels: pd.DataFrame,
    task: str,
    tune_min: int,
    tune_max: int,
    names: list[str],
):
    tune_qrels = qrels[qrels["query_id"].map(query_number).between(tune_min, tune_max)]
    best = None
    best_weights = None
    for weights in composition_weights(names, step=0.1):
        model_name = f"{task}_split_weighted"
        run = fuse_runs({name: runs[name] for name in names}, weights, model_name, task, 10)
        _, summary = evaluate_run(restrict_run(run, tune_min, tune_max), tune_qrels, k=10)
        if best is None or summary["map"] > best["map"]:
            best = summary
            best_weights = weights
    final = fuse_runs({name: runs[name] for name in names}, best_weights, f"{task}_split_weighted", task, 10)
    return final, best_weights, best


def positive_counts(qrels: pd.DataFrame):
    return qrels[qrels["relevance"].astype(int) > 0].groupby("query_id")["doc_id"].nunique()


def current_error_analysis(bundle, case_qrels, statute_qrels):
    case_run = pd.read_csv(OUT_DIR / "case_best_tuned.csv")
    statute_run = pd.read_csv(OUT_DIR / "statute_best_tuned.csv")
    case_details, case_summary = summarize(case_run, case_qrels, "case", "case_best_tuned_recomputed")
    statute_details, statute_summary = summarize(statute_run, statute_qrels, "statute", "statute_best_tuned_recomputed")

    cases = bundle.cases.set_index("doc_id")["text"].to_dict()
    queries = bundle.queries.set_index("query_id")["text"].to_dict()
    rel = case_qrels[case_qrels["relevance"].astype(int) > 0]
    rows = []
    for query_id in case_details.sort_values(["average_precision", "ndcg"]).head(10)["query_id"]:
        relevant = rel[rel["query_id"] == query_id]["doc_id"].astype(str).tolist()
        retrieved = (
            case_run[case_run["query_id"] == query_id]
            .sort_values("rank")
            .head(10)["doc_id"]
            .astype(str)
            .tolist()
        )
        rows.append(
            {
                "query_id": query_id,
                "query_text": queries.get(query_id, "")[:800],
                "relevant_docs": " ".join(relevant[:12]),
                "relevant_titles": " | ".join(cases.get(doc_id, "").split("\n")[0][:120] for doc_id in relevant[:5]),
                "top10_docs": " ".join(retrieved),
                "top10_titles": " | ".join(cases.get(doc_id, "").split("\n")[0][:120] for doc_id in retrieved[:5]),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "case_error_analysis_worst_queries.csv", index=False)
    return case_details, case_summary, statute_details, statute_summary


def corpus_profile(bundle):
    profile = {
        "num_queries": int(len(bundle.queries)),
        "num_cases": int(len(bundle.cases)),
        "num_statutes": int(len(bundle.statutes)),
        "query_words": bundle.queries["text"].str.split().str.len().describe().to_dict(),
        "case_words": bundle.cases["text"].str.split().str.len().describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_dict(),
        "statute_words": bundle.statutes["text"].str.split().str.len().describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_dict(),
        "query_samples": bundle.queries[["query_id", "text"]].head(10).to_dict(orient="records"),
        "case_samples": bundle.cases[["doc_id", "text"]].sample(8, random_state=7).to_dict(orient="records"),
        "statute_samples": bundle.statutes[["doc_id", "text"]].sample(8, random_state=7).to_dict(orient="records"),
    }
    (OUT_DIR / "corpus_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def add_experiment(rows, name, change, hypothesis, case_summary=None, statute_summary=None, notes=""):
    case_after = case_summary or BASELINE["case"]
    statute_after = statute_summary or BASELINE["statute"]
    rows.append(
        {
            "experiment": name,
            "change_made": change,
            "hypothesis": hypothesis,
            "case_map_before": BASELINE["case"]["map"],
            "case_map_after": case_after["map"],
            "statute_map_before": BASELINE["statute"]["map"],
            "statute_map_after": statute_after["map"],
            "case_ndcg10_before": BASELINE["case"]["ndcg_at_k"],
            "case_ndcg10_after": case_after["ndcg_at_k"],
            "statute_ndcg10_before": BASELINE["statute"]["ndcg_at_k"],
            "statute_ndcg10_after": statute_after["ndcg_at_k"],
            "notes": notes,
        }
    )


def write_markdown_report(profile, error_stats, experiments, split_notes):
    lines = [
        "# FIRE 2019 AILA Experiment Report",
        "",
        "## Corpus and Failure Analysis",
        f"- Queries: {profile['num_queries']}; cases: {profile['num_cases']}; statutes: {profile['num_statutes']}.",
        f"- Query length mean/median: {profile['query_words']['mean']:.1f}/{profile['query_words']['50%']:.1f} words.",
        f"- Case length mean/median/95th: {profile['case_words']['mean']:.1f}/{profile['case_words']['50%']:.1f}/{profile['case_words']['95%']:.1f} words.",
        f"- Statute length mean/median/95th: {profile['statute_words']['mean']:.1f}/{profile['statute_words']['50%']:.1f}/{profile['statute_words']['95%']:.1f} words.",
        f"- Current tuned case run has {error_stats['case_zero_top10']} of 40 eval queries with zero relevant cases in top 10.",
        f"- Current tuned statute run has {error_stats['statute_zero_top10']} of 40 eval queries with zero relevant statutes in top 10.",
        "- Main challenge: narrative fact patterns are long and mixed with procedural posture, while cases are full judgments. Lexical systems match surface terms, but full-document scoring dilutes the short legally decisive passages.",
        "",
        "## Evaluation Integrity",
        "- Existing `tune_fusion.py` optimizes and reports on Q11-Q50, so its best tuned numbers are useful for diagnosis but optimistic.",
        f"- Split-safe fusion in this report tunes on Q11-Q30 and reports test metrics on Q31-Q50. {split_notes}",
        "- `evaluation/metrics.py` was not modified.",
        "",
        "## Experiments",
    ]
    for row in experiments:
        lines.extend(
            [
                f"### Experiment: {row['experiment']}",
                f"- Change made: {row['change_made']}",
                f"- Hypothesis: {row['hypothesis']}",
                f"- Case MAP: {row['case_map_before']:.4f} -> {row['case_map_after']:.4f}",
                f"- Statute MAP: {row['statute_map_before']:.4f} -> {row['statute_map_after']:.4f}",
                f"- Case nDCG@10: {row['case_ndcg10_before']:.4f} -> {row['case_ndcg10_after']:.4f}",
                f"- Statute nDCG@10: {row['statute_ndcg10_before']:.4f} -> {row['statute_ndcg10_after']:.4f}",
                f"- Notes: {row['notes']}",
                "",
            ]
        )
    (OUT_DIR / "legal_ir_experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    profile = corpus_profile(bundle)
    eval_sets = normalize_queries_qrels(bundle, 11)
    case_qrels = eval_sets["case"]
    statute_qrels = eval_sets["statute"]
    case_details, case_summary, statute_details, statute_summary = current_error_analysis(bundle, case_qrels, statute_qrels)

    error_stats = {
        "case_zero_top10": int((case_details["precision"] == 0).sum()),
        "statute_zero_top10": int((statute_details["precision"] == 0).sum()),
    }
    experiments = []

    case_runs = {}
    case_tfidf = TfidfRetriever("case_lex_tfidf", model_dir=MODEL_DIR).fit(bundle.cases)
    case_runs["tfidf"] = case_tfidf.retrieve_many(bundle.queries, args.candidate_k, task="case")
    default_bm25 = BM25Retriever("case_bm25_default", model_dir=MODEL_DIR).fit(bundle.cases)
    case_runs["bm25_default"] = default_bm25.retrieve_many(bundle.queries, args.candidate_k, task="case")

    bm25_rows = []
    best_bm25 = None
    for k1 in [0.5, 1.0, 1.5, 2.0]:
        for b in [0.5, 0.75, 1.0]:
            name = f"case_bm25_k1_{str(k1).replace('.', '_')}_b_{str(b).replace('.', '_')}"
            run_df = BM25Retriever(name, model_dir=MODEL_DIR, k1=k1, b=b).fit(bundle.cases).retrieve_many(bundle.queries, 10, task="case")
            _, summary = summarize(run_df, case_qrels, "case", name)
            bm25_rows.append({**summary, "k1": k1, "b": b})
            if best_bm25 is None or summary["map"] > best_bm25[1]["map"]:
                best_bm25 = (name, summary, k1, b)
    pd.DataFrame(bm25_rows).sort_values("map", ascending=False).to_csv(OUT_DIR / "case_bm25_param_grid.csv", index=False)
    tuned_name, tuned_summary, tuned_k1, tuned_b = best_bm25
    tuned_candidate = BM25Retriever(tuned_name, model_dir=MODEL_DIR, k1=tuned_k1, b=tuned_b).fit(bundle.cases).retrieve_many(bundle.queries, args.candidate_k, task="case")
    case_runs["bm25_tuned"] = tuned_candidate
    save_run(tuned_candidate.sort_values(["query_id", "rank"]).groupby("query_id").head(10), tuned_name)
    add_experiment(
        experiments,
        "BM25 parameter tuning (cases)",
        f"Grid searched k1 in [0.5, 1.0, 1.5, 2.0] and b in [0.5, 0.75, 1.0]; best k1={tuned_k1}, b={tuned_b}.",
        "Legal judgments are long; lower or higher length normalization can change whether decisive passages are drowned by document length.",
        case_summary=tuned_summary,
        notes="This is a single-model diagnostic on Q11-Q50, not a final honest tuned score.",
    )

    passage_rows = []
    best_passage = None
    for chunk_size in [180, 320, 500]:
        for aggregate in ["max", "sum"]:
            name = f"case_passage_bm25_c{chunk_size}_{aggregate}"
            retriever = PassageBM25Retriever(
                name,
                model_dir=MODEL_DIR,
                k1=tuned_k1,
                b=tuned_b,
                chunk_size=chunk_size,
                overlap=max(40, chunk_size // 4),
                aggregate=aggregate,
            ).fit(bundle.cases)
            run_df = retriever.retrieve_many(bundle.queries, 10, task="case")
            _, summary = summarize(run_df, case_qrels, "case", name)
            passage_rows.append({**summary, "chunk_size": chunk_size, "aggregate": aggregate})
            if best_passage is None or summary["map"] > best_passage[1]["map"]:
                best_passage = (name, summary, chunk_size, aggregate)
    pd.DataFrame(passage_rows).sort_values("map", ascending=False).to_csv(OUT_DIR / "case_passage_bm25_grid.csv", index=False)
    passage_name, passage_summary, passage_chunk, passage_agg = best_passage
    passage_candidate = PassageBM25Retriever(
        passage_name,
        model_dir=MODEL_DIR,
        k1=tuned_k1,
        b=tuned_b,
        chunk_size=passage_chunk,
        overlap=max(40, passage_chunk // 4),
        aggregate=passage_agg,
    ).fit(bundle.cases).retrieve_many(bundle.queries, args.candidate_k, task="case")
    case_runs["passage_bm25"] = passage_candidate
    save_run(passage_candidate.sort_values(["query_id", "rank"]).groupby("query_id").head(10), passage_name)
    add_experiment(
        experiments,
        "Passage-level BM25 (cases)",
        f"Chunked cases into {passage_chunk}-token windows and aggregated passage scores with {passage_agg}.",
        "Long full judgments dilute local evidence; matching the query against passages should surface the relevant legal/factual portion.",
        case_summary=passage_summary,
        notes="Uses only local lexical retrieval, so it is fast and does not require downloading new models.",
    )

    rrf_case = rrf_fuse(
        {"tfidf": case_runs["tfidf"], "bm25": case_runs["bm25_tuned"], "passage": case_runs["passage_bm25"]},
        "case_rrf_lexical_passage",
        "case",
        10,
    )
    save_run(rrf_case, "case_rrf_lexical_passage")
    _, rrf_case_summary = summarize(rrf_case, case_qrels, "case", "case_rrf_lexical_passage")
    add_experiment(
        experiments,
        "RRF lexical+passage (cases)",
        "Combined TF-IDF, tuned BM25, and passage BM25 with Reciprocal Rank Fusion.",
        "Rank-level fusion should be robust when score scales differ and a document is found by only one strong signal.",
        case_summary=rrf_case_summary,
        notes="Parameter-free fusion baseline; no Q11-Q50 tuning.",
    )

    split_case_run, split_case_weights, split_case_val = tune_weighted_fusion(
        {"tfidf": case_runs["tfidf"], "bm25": case_runs["bm25_tuned"], "passage": case_runs["passage_bm25"]},
        case_qrels,
        "case",
        11,
        30,
        ["tfidf", "bm25", "passage"],
    )
    save_run(split_case_run, "case_split_weighted_lexical_passage")
    case_test_qrels = case_qrels[case_qrels["query_id"].map(query_number).between(31, 50)]
    _, split_case_test_summary = evaluate_run(restrict_run(split_case_run, 31, 50), case_test_qrels, k=10)
    split_case_test_summary.update({"task": "case", "model": "case_split_weighted_lexical_passage", "k": 10})

    statute_runs = {}
    statute_runs["tfidf"] = TfidfRetriever("statute_lex_tfidf", model_dir=MODEL_DIR).fit(bundle.statutes).retrieve_many(bundle.queries, args.candidate_k, task="statute")
    statute_runs["bm25"] = BM25Retriever("statute_bm25_default", model_dir=MODEL_DIR).fit(bundle.statutes).retrieve_many(bundle.queries, args.candidate_k, task="statute")
    train_queries = filter_queries(bundle.queries, max_id=10)
    train_qrels = filter_qrels_to_queries(bundle.statute_qrels, train_queries)
    statute_runs["classifier"] = StatuteClassifierRetriever("statute_split_classifier", model_dir=MODEL_DIR).fit(train_queries, train_qrels).retrieve_many(bundle.queries, args.candidate_k, task="statute")
    rrf_statute = rrf_fuse(statute_runs, "statute_rrf_classifier_tfidf_bm25", "statute", 10)
    save_run(rrf_statute, "statute_rrf_classifier_tfidf_bm25")
    _, rrf_statute_summary = summarize(rrf_statute, statute_qrels, "statute", "statute_rrf_classifier_tfidf_bm25")
    add_experiment(
        experiments,
        "RRF classifier+lexical (statutes)",
        "Combined the supervised statute classifier, TF-IDF, and BM25 with Reciprocal Rank Fusion.",
        "The classifier is strong but sparse; RRF can recover statutes ranked well by lexical match without score calibration.",
        statute_summary=rrf_statute_summary,
        notes="Classifier training remains restricted to Q1-Q10.",
    )

    split_statute_run, split_statute_weights, split_statute_val = tune_weighted_fusion(
        statute_runs,
        statute_qrels,
        "statute",
        11,
        30,
        ["classifier", "tfidf", "bm25"],
    )
    save_run(split_statute_run, "statute_split_weighted_classifier_tfidf_bm25")
    statute_test_qrels = statute_qrels[statute_qrels["query_id"].map(query_number).between(31, 50)]
    _, split_statute_test_summary = evaluate_run(restrict_run(split_statute_run, 31, 50), statute_test_qrels, k=10)
    split_statute_test_summary.update({"task": "statute", "model": "statute_split_weighted_classifier_tfidf_bm25", "k": 10})

    current_case_test = pd.read_csv(OUT_DIR / "case_best_tuned.csv")
    current_statute_test = pd.read_csv(OUT_DIR / "statute_best_tuned.csv")
    _, current_case_test_summary = evaluate_run(restrict_run(current_case_test, 31, 50), case_test_qrels, k=10)
    _, current_statute_test_summary = evaluate_run(restrict_run(current_statute_test, 31, 50), statute_test_qrels, k=10)
    split_notes = (
        f"Case weights={split_case_weights}, val MAP={split_case_val['map']:.4f}, "
        f"test MAP {current_case_test_summary['map']:.4f} -> {split_case_test_summary['map']:.4f}. "
        f"Statute weights={split_statute_weights}, val MAP={split_statute_val['map']:.4f}, "
        f"test MAP {current_statute_test_summary['map']:.4f} -> {split_statute_test_summary['map']:.4f}."
    )
    add_experiment(
        experiments,
        "Split-safe weighted fusion",
        "Tuned fusion weights on Q11-Q30 and evaluated once on held-out Q31-Q50.",
        "Separating validation from test gives an honest estimate and shows whether tuned weights generalize.",
        case_summary=split_case_test_summary,
        statute_summary=split_statute_test_summary,
        notes=split_notes,
    )

    pd.DataFrame(experiments).to_csv(OUT_DIR / "legal_ir_experiment_log.csv", index=False)
    write_markdown_report(profile, error_stats, experiments, split_notes)
    print(pd.DataFrame(experiments)[["experiment", "case_map_after", "statute_map_after", "case_ndcg10_after", "statute_ndcg10_after"]].to_string(index=False))
    print(f"Report written to {OUT_DIR / 'legal_ir_experiment_report.md'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Focused FIRE 2019 AILA error analysis and local retrieval experiments")
    parser.add_argument("--candidate-k", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
