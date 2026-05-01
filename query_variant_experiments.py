from pathlib import Path

import pandas as pd

from data_loader import AilaDataLoader
from evaluation.metrics import evaluate_run
from main import add_clean_columns, filter_qrels_to_queries, filter_queries
from retrieval import BM25Retriever, TfidfRetriever


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs" / "experiments"

LEGAL_CUES = {
    "accused",
    "acquittal",
    "appeal",
    "appellant",
    "arbitration",
    "bail",
    "charge",
    "civil",
    "compensation",
    "constitution",
    "contract",
    "conviction",
    "court",
    "criminal",
    "custody",
    "damages",
    "death",
    "decree",
    "delay",
    "dismissed",
    "evidence",
    "fir",
    "high",
    "ipc",
    "investigation",
    "judge",
    "judgment",
    "jurisdiction",
    "land",
    "leave",
    "liability",
    "murder",
    "offence",
    "order",
    "petition",
    "police",
    "prosecution",
    "respondent",
    "rights",
    "section",
    "sentence",
    "service",
    "sessions",
    "statutory",
    "trial",
    "witness",
    "writ",
}


def load_bundle():
    bundle = AilaDataLoader(ROOT).load_processed()
    if "clean_text" not in bundle.queries.columns:
        bundle = add_clean_columns(bundle)
    for frame in [bundle.queries, bundle.cases, bundle.statutes]:
        frame["clean_text"] = frame["clean_text"].fillna("").astype(str)
    return bundle


def make_query_variants(queries: pd.DataFrame) -> pd.DataFrame:
    queries = queries.copy()
    token_lists = queries["clean_text"].str.split()
    for n in [80, 120, 180, 250, 350]:
        queries[f"first_{n}"] = token_lists.map(lambda toks, n=n: " ".join(toks[:n]))
        queries[f"last_{n}"] = token_lists.map(lambda toks, n=n: " ".join(toks[-n:]))
    queries["first_last_120"] = token_lists.map(lambda toks: " ".join(toks[:120] + toks[-120:]))
    queries["legal_cue_window"] = token_lists.map(legal_cue_window)
    return queries


def legal_cue_window(tokens: list[str]) -> str:
    keep = []
    for idx, token in enumerate(tokens):
        if token in LEGAL_CUES or token.startswith("section"):
            keep.extend(tokens[max(0, idx - 5) : idx + 8])
    if not keep:
        keep = tokens[:180]
    return " ".join(keep[:350])


def save_run(run: pd.DataFrame, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run.to_csv(OUT_DIR / f"{name}.csv", index=False)


def main():
    bundle = load_bundle()
    bundle.queries = make_query_variants(bundle.queries)
    eval_queries = filter_queries(bundle.queries, min_id=11)
    case_qrels = filter_qrels_to_queries(bundle.case_qrels, eval_queries)
    statute_qrels = filter_qrels_to_queries(bundle.statute_qrels, eval_queries)

    case_tfidf = TfidfRetriever("case_query_variant_tfidf", model_dir=MODEL_DIR).fit(bundle.cases)
    case_bm25 = BM25Retriever("case_query_variant_bm25", model_dir=MODEL_DIR, k1=1.5, b=1.0).fit(bundle.cases)
    statute_tfidf = TfidfRetriever("statute_query_variant_tfidf", model_dir=MODEL_DIR).fit(bundle.statutes)
    statute_bm25 = BM25Retriever("statute_query_variant_bm25", model_dir=MODEL_DIR).fit(bundle.statutes)

    rows = []
    query_cols = ["clean_text", "first_80", "first_120", "first_180", "first_250", "first_350", "last_120", "last_180", "first_last_120", "legal_cue_window"]
    for query_col in query_cols:
        for task, retrievers, qrels in [
            ("case", {"tfidf": case_tfidf, "bm25": case_bm25}, case_qrels),
            ("statute", {"tfidf": statute_tfidf, "bm25": statute_bm25}, statute_qrels),
        ]:
            for kind, retriever in retrievers.items():
                model = f"{task}_{kind}_{query_col}"
                run = retriever.retrieve_many(bundle.queries, top_k=10, query_col=query_col, task=task)
                _, summary = evaluate_run(run, qrels, k=10)
                rows.append({**summary, "task": task, "model": model, "query_col": query_col, "retriever": kind})
                save_run(run, model)

    board = pd.DataFrame(rows).sort_values(["task", "map"], ascending=[True, False])
    board.to_csv(OUT_DIR / "query_variant_leaderboard.csv", index=False)
    print(board.groupby("task").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
