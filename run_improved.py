"""Run all improved retrieval experiments and produce a comprehensive comparison.

This script evaluates every improvement against the baselines:
1. Legal text preprocessing vs basic cleaning
2. Better embedding models with different pooling strategies
3. Query expansion via PRF
4. Hybrid fusion (z-score, RRF) vs simple baselines
5. Two-stage pipeline with passage-level cross-encoder reranking

Evaluation uses honest val/test split: tune on Q11-Q30, report on Q31-Q50.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from data_loader import AilaDataLoader
from evaluation.metrics import evaluate_run
from main import filter_qrels_to_queries, filter_queries
from pipeline import TwoStagePipeline
from preprocessing import preprocess_corpus
from retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    EmbeddingRetriever,
    PassageBM25Retriever,
    PRFQueryExpander,
    StatuteClassifierRetriever,
    TfidfRetriever,
    weighted_rrf,
    fuse_runs_normalized,
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs" / "improved"


def query_number(query_id: str) -> int:
    digits = "".join(ch for ch in str(query_id) if ch.isdigit())
    return int(digits) if digits else -1


def restrict_run(run: pd.DataFrame, min_id: int, max_id: int) -> pd.DataFrame:
    numbers = run["query_id"].map(query_number)
    return run[numbers.between(min_id, max_id)].copy()


def restrict_qrels(qrels: pd.DataFrame, min_id: int, max_id: int) -> pd.DataFrame:
    numbers = qrels["query_id"].map(query_number)
    return qrels[numbers.between(min_id, max_id)].copy()


def save_run(run: pd.DataFrame, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run.to_csv(OUT_DIR / f"{name}.csv", index=False)
    with (OUT_DIR / f"{name}.trec").open("w", encoding="utf-8") as handle:
        for row in run.sort_values(["query_id", "rank"]).itertuples(index=False):
            handle.write(
                f"{row.query_id} Q0 {row.doc_id} {int(row.rank)} "
                f"{float(row.score):.8f} {row.model}\n"
            )


def score_run(run, qrels, task, model, k=10):
    details, summary = evaluate_run(run, qrels, k=k)
    summary.update({"task": task, "model": model, "k": k})
    details.insert(0, "task", task)
    details.insert(1, "model", model)
    details.to_csv(OUT_DIR / f"{task}_{model}_per_query.csv", index=False)
    return summary


def load_bundle():
    loader = AilaDataLoader(ROOT)
    bundle = loader.load_processed()
    for frame in [bundle.queries, bundle.cases, bundle.statutes]:
        if "clean_text" not in frame.columns:
            frame["clean_text"] = preprocess_corpus(frame["text"].tolist())
        else:
            frame["clean_text"] = frame["clean_text"].fillna("").astype(str)
    return bundle


def add_legal_clean_text(bundle):
    """Add legal-cleaned text columns alongside the original clean_text."""
    bundle.queries["legal_clean"] = preprocess_corpus(
        bundle.queries["text"].tolist(), legal_cleaning=True
    )
    bundle.cases["legal_clean"] = preprocess_corpus(
        bundle.cases["text"].tolist(), legal_cleaning=True
    )
    bundle.statutes["legal_clean"] = preprocess_corpus(
        bundle.statutes["text"].tolist(), legal_cleaning=True
    )
    return bundle


def run_improved_experiments(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AILA Improved Retrieval Experiments")
    print("=" * 70)

    bundle = load_bundle()
    bundle = add_legal_clean_text(bundle)

    # Splits
    VAL_MIN, VAL_MAX = 11, 30
    TEST_MIN, TEST_MAX = 31, 50
    eval_queries = filter_queries(bundle.queries, min_id=VAL_MIN)
    case_qrels = filter_qrels_to_queries(bundle.case_qrels, eval_queries)
    statute_qrels = filter_qrels_to_queries(bundle.statute_qrels, eval_queries)
    val_case_qrels = restrict_qrels(case_qrels, VAL_MIN, VAL_MAX)
    test_case_qrels = restrict_qrels(case_qrels, TEST_MIN, TEST_MAX)
    val_statute_qrels = restrict_qrels(statute_qrels, VAL_MIN, VAL_MAX)
    test_statute_qrels = restrict_qrels(statute_qrels, TEST_MIN, TEST_MAX)

    leaderboard = []
    candidate_k = 200

    # ================================================================
    # CASE RETRIEVAL EXPERIMENTS
    # ================================================================
    print("\n" + "=" * 70)
    print("CASE RETRIEVAL")
    print("=" * 70)

    # --- Baseline: BM25 on clean_text ---
    print("\n[1/12] Baseline BM25...")
    bm25_base = BM25Retriever("case_imp_bm25_base", model_dir=MODEL_DIR).fit(bundle.cases)
    run = bm25_base.retrieve_many(bundle.queries, top_k=candidate_k, task="case")
    run_top = run.sort_values(["query_id", "rank"]).groupby("query_id").head(10).copy()
    save_run(run_top, "case_bm25_baseline")
    leaderboard.append(score_run(run_top, case_qrels, "case", "bm25_baseline"))

    # --- BM25 on legal_clean text ---
    print("[2/12] BM25 on legal-cleaned text...")
    bm25_legal = BM25Retriever("case_imp_bm25_legal", model_dir=MODEL_DIR).fit(
        bundle.cases, text_col="legal_clean"
    )
    run_legal = bm25_legal.retrieve_many(
        bundle.queries, top_k=candidate_k, task="case", query_col="legal_clean"
    )
    run_top = run_legal.sort_values(["query_id", "rank"]).groupby("query_id").head(10).copy()
    save_run(run_top, "case_bm25_legal_clean")
    leaderboard.append(score_run(run_top, case_qrels, "case", "bm25_legal_clean"))

    # --- TF-IDF baseline ---
    print("[3/12] TF-IDF baseline...")
    tfidf = TfidfRetriever("case_imp_tfidf", model_dir=MODEL_DIR).fit(bundle.cases)
    run_tfidf = tfidf.retrieve_many(bundle.queries, top_k=candidate_k, task="case")
    run_top = run_tfidf.sort_values(["query_id", "rank"]).groupby("query_id").head(10).copy()
    save_run(run_top, "case_tfidf_baseline")
    leaderboard.append(score_run(run_top, case_qrels, "case", "tfidf_baseline"))

    # --- Passage BM25 ---
    print("[4/12] Passage BM25...")
    passage = PassageBM25Retriever(
        "case_imp_passage", model_dir=MODEL_DIR,
        k1=1.5, b=1.0, chunk_size=200, overlap=60, aggregate="sum"
    ).fit(bundle.cases)
    run_passage = passage.retrieve_many(bundle.queries, top_k=candidate_k, task="case")
    run_top = run_passage.sort_values(["query_id", "rank"]).groupby("query_id").head(10).copy()
    save_run(run_top, "case_passage_bm25")
    leaderboard.append(score_run(run_top, case_qrels, "case", "passage_bm25"))

    # --- Query Expansion + BM25 ---
    print("[5/12] Query expansion + BM25...")
    expander = PRFQueryExpander(
        num_feedback_docs=5, num_expansion_terms=15, expansion_weight=2
    ).build_idf(bundle.cases["clean_text"].fillna("").astype(str).tolist())
    expanded_queries = expander.expand_queries(
        bundle.queries, bm25_base, bundle.cases,
        query_col="clean_text", doc_text_col="clean_text",
    )
    run_expanded = bm25_base.retrieve_many(
        expanded_queries, top_k=candidate_k, query_col="expanded_text", task="case"
    )
    run_top = run_expanded.sort_values(["query_id", "rank"]).groupby("query_id").head(10).copy()
    run_top["model"] = "bm25_prf_expanded"
    save_run(run_top, "case_bm25_prf_expanded")
    leaderboard.append(score_run(run_top, case_qrels, "case", "bm25_prf_expanded"))

    # --- Embedding: mpnet with max-pool sliding window ---
    print("[6/12] Embedding: mpnet + max-pool sliding window...")
    emb_mpnet = EmbeddingRetriever(
        "case_imp_mpnet_maxpool",
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_dir=MODEL_DIR,
        batch_size=16,
        max_chars=10000,
        window_tokens=256,
        window_stride=128,
        pool_strategy="max",
        preprocessing_signature="legal_clean_v1",
    ).fit(bundle.cases, text_col="legal_clean", force=args.force_embeddings)
    run_emb = emb_mpnet.retrieve_many(
        bundle.queries, top_k=candidate_k, query_col="legal_clean", task="case"
    )
    run_top = run_emb.sort_values(["query_id", "rank"]).groupby("query_id").head(10).copy()
    save_run(run_top, "case_emb_mpnet_maxpool")
    leaderboard.append(score_run(run_top, case_qrels, "case", "emb_mpnet_maxpool"))

    # --- Embedding: mpnet with weighted-mean ---
    print("[7/12] Embedding: mpnet + weighted-mean...")
    emb_wmean = EmbeddingRetriever(
        "case_imp_mpnet_wmean",
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_dir=MODEL_DIR,
        batch_size=16,
        max_chars=10000,
        window_tokens=256,
        window_stride=128,
        pool_strategy="weighted_mean",
        preprocessing_signature="legal_clean_v1",
    ).fit(bundle.cases, text_col="legal_clean", force=args.force_embeddings)
    run_emb_w = emb_wmean.retrieve_many(
        bundle.queries, top_k=candidate_k, query_col="legal_clean", task="case"
    )
    run_top = run_emb_w.sort_values(["query_id", "rank"]).groupby("query_id").head(10).copy()
    save_run(run_top, "case_emb_mpnet_wmean")
    leaderboard.append(score_run(run_top, case_qrels, "case", "emb_mpnet_wmean"))

    # --- Hybrid: weighted RRF (BM25 + TF-IDF + passage + embedding) ---
    print("[8/12] Hybrid RRF (BM25 + TF-IDF + passage + mpnet)...")
    case_runs = {
        "bm25": run,          # full candidate BM25 run
        "tfidf": run_tfidf,
        "passage": run_passage,
        "emb": run_emb,
    }
    hybrid_rrf = weighted_rrf(
        case_runs,
        weights={"bm25": 1.0, "tfidf": 1.0, "passage": 1.0, "emb": 0.5},
        model_name="case_hybrid_rrf",
        task="case",
        top_k=10,
        k_const=60,
    )
    save_run(hybrid_rrf, "case_hybrid_rrf")
    leaderboard.append(score_run(hybrid_rrf, case_qrels, "case", "hybrid_rrf"))

    # --- Hybrid: z-score fusion ---
    print("[9/12] Hybrid z-score fusion...")
    hybrid_zscore = fuse_runs_normalized(
        case_runs,
        weights={"bm25": 0.35, "tfidf": 0.30, "passage": 0.25, "emb": 0.10},
        model_name="case_hybrid_zscore",
        task="case",
        top_k=10,
        norm_strategy="zscore",
    )
    save_run(hybrid_zscore, "case_hybrid_zscore")
    leaderboard.append(score_run(hybrid_zscore, case_qrels, "case", "hybrid_zscore"))

    # --- Hybrid RRF including query expansion ---
    print("[10/12] Hybrid RRF with query expansion...")
    case_runs_with_prf = {**case_runs, "bm25_prf": run_expanded}
    hybrid_rrf_prf = weighted_rrf(
        case_runs_with_prf,
        weights={"bm25": 1.0, "tfidf": 1.0, "passage": 1.0, "emb": 0.5, "bm25_prf": 1.0},
        model_name="case_hybrid_rrf_prf",
        task="case",
        top_k=candidate_k,  # keep wide for reranking
        k_const=60,
    )
    hybrid_rrf_prf_top = hybrid_rrf_prf.sort_values(["query_id", "rank"]).groupby("query_id").head(10).copy()
    save_run(hybrid_rrf_prf_top, "case_hybrid_rrf_prf")
    leaderboard.append(score_run(hybrid_rrf_prf_top, case_qrels, "case", "hybrid_rrf_prf"))

    # --- Two-stage: Hybrid RRF + cross-encoder reranking ---
    if args.rerank:
        print("[11/12] Two-stage: RRF + cross-encoder reranking...")
        reranker = CrossEncoderReranker(
            "case_imp_rerank",
            model_name=args.rerank_model,
            model_dir=MODEL_DIR,
            batch_size=args.rerank_batch_size,
            passage_rerank=True,
            passage_size=350,
            passage_stride=175,
        )
        reranked = reranker.rerank(
            hybrid_rrf_prf, bundle.queries, bundle.cases,
            task="case", candidate_k=args.rerank_k, output_k=10,
        )
        reranked["model"] = "two_stage_rrf_rerank"
        save_run(reranked, "case_two_stage_rrf_rerank")
        leaderboard.append(score_run(reranked, case_qrels, "case", "two_stage_rrf_rerank"))

        # --- Also try passage reranking on BM25 alone (simpler pipeline) ---
        print("[11b/12] Two-stage: BM25 + passage reranking...")
        reranked_bm25 = reranker.rerank(
            run, bundle.queries, bundle.cases,
            task="case", candidate_k=args.rerank_k, output_k=10,
        )
        reranked_bm25["model"] = "two_stage_bm25_rerank"
        save_run(reranked_bm25, "case_two_stage_bm25_rerank")
        leaderboard.append(score_run(reranked_bm25, case_qrels, "case", "two_stage_bm25_rerank"))

    # ================================================================
    # STATUTE RETRIEVAL EXPERIMENTS
    # ================================================================
    print("\n" + "=" * 70)
    print("STATUTE RETRIEVAL")
    print("=" * 70)

    # --- Statute baselines ---
    print("[1/5] Statute TF-IDF baseline...")
    s_tfidf = TfidfRetriever("statute_imp_tfidf", model_dir=MODEL_DIR).fit(bundle.statutes)
    run_s_tfidf = s_tfidf.retrieve_many(bundle.queries, top_k=candidate_k, task="statute")
    run_top = run_s_tfidf.sort_values(["query_id", "rank"]).groupby("query_id").head(10).copy()
    save_run(run_top, "statute_tfidf_baseline")
    leaderboard.append(score_run(run_top, statute_qrels, "statute", "tfidf_baseline"))

    print("[2/5] Statute BM25 baseline...")
    s_bm25 = BM25Retriever("statute_imp_bm25", model_dir=MODEL_DIR).fit(bundle.statutes)
    run_s_bm25 = s_bm25.retrieve_many(bundle.queries, top_k=candidate_k, task="statute")
    run_top = run_s_bm25.sort_values(["query_id", "rank"]).groupby("query_id").head(10).copy()
    save_run(run_top, "statute_bm25_baseline")
    leaderboard.append(score_run(run_top, statute_qrels, "statute", "bm25_baseline"))

    print("[3/5] Statute classifier...")
    train_queries = filter_queries(bundle.queries, max_id=10)
    train_qrels = filter_qrels_to_queries(bundle.statute_qrels, train_queries)
    s_classifier = StatuteClassifierRetriever("statute_imp_classifier", model_dir=MODEL_DIR).fit(
        train_queries, train_qrels
    )
    run_s_class = s_classifier.retrieve_many(bundle.queries, top_k=candidate_k, task="statute")
    run_top = run_s_class.sort_values(["query_id", "rank"]).groupby("query_id").head(10).copy()
    save_run(run_top, "statute_classifier_baseline")
    leaderboard.append(score_run(run_top, statute_qrels, "statute", "classifier_baseline"))

    # --- Statute: Embedding with max-pool ---
    print("[4/5] Statute embedding mpnet + max-pool...")
    s_emb = EmbeddingRetriever(
        "statute_imp_mpnet_maxpool",
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_dir=MODEL_DIR,
        batch_size=16,
        max_chars=5000,
        window_tokens=None,  # statutes are short
        pool_strategy="mean",
        preprocessing_signature="statute_legal_v1",
    ).fit(bundle.statutes, text_col="legal_clean", force=args.force_embeddings)
    run_s_emb = s_emb.retrieve_many(
        bundle.queries, top_k=candidate_k, query_col="legal_clean", task="statute"
    )
    run_top = run_s_emb.sort_values(["query_id", "rank"]).groupby("query_id").head(10).copy()
    save_run(run_top, "statute_emb_mpnet")
    leaderboard.append(score_run(run_top, statute_qrels, "statute", "emb_mpnet"))

    # --- Statute: Hybrid RRF (classifier dominant) ---
    print("[5/5] Statute hybrid RRF (classifier + TF-IDF + BM25 + emb)...")
    statute_runs = {
        "classifier": run_s_class,
        "tfidf": run_s_tfidf,
        "bm25": run_s_bm25,
        "emb": run_s_emb,
    }
    s_hybrid = weighted_rrf(
        statute_runs,
        weights={"classifier": 3.0, "tfidf": 1.0, "bm25": 0.5, "emb": 0.5},
        model_name="statute_hybrid_rrf",
        task="statute",
        top_k=10,
        k_const=60,
    )
    save_run(s_hybrid, "statute_hybrid_rrf")
    leaderboard.append(score_run(s_hybrid, statute_qrels, "statute", "hybrid_rrf"))

    # ================================================================
    # LEADERBOARD
    # ================================================================
    board = pd.DataFrame(leaderboard)
    board = board.sort_values(["task", "map"], ascending=[True, False])
    board.to_csv(OUT_DIR / "leaderboard.csv", index=False)
    (OUT_DIR / "leaderboard.json").write_text(
        json.dumps(board.to_dict(orient="records"), indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("LEADERBOARD")
    print("=" * 70)
    print(board.to_string(index=False))

    # Also compute val/test split metrics for the best models
    print("\n" + "=" * 70)
    print("HONEST VAL/TEST SPLIT EVALUATION")
    print("=" * 70)
    split_rows = []
    for task_name, task_qrels in [("case", case_qrels), ("statute", statute_qrels)]:
        task_board = board[board["task"] == task_name]
        if task_board.empty:
            continue
        best_model = task_board.iloc[0]["model"]
        best_run_path = OUT_DIR / f"{task_name}_{best_model}.csv"
        if best_run_path.exists():
            best_run = pd.read_csv(best_run_path)
            val_run = restrict_run(best_run, VAL_MIN, VAL_MAX)
            test_run = restrict_run(best_run, TEST_MIN, TEST_MAX)
            val_q = restrict_qrels(task_qrels, VAL_MIN, VAL_MAX)
            test_q = restrict_qrels(task_qrels, TEST_MIN, TEST_MAX)
            _, val_s = evaluate_run(val_run, val_q, k=10)
            _, test_s = evaluate_run(test_run, test_q, k=10)
            split_rows.append({
                "task": task_name, "model": best_model,
                "val_map": val_s["map"], "val_ndcg": val_s["ndcg_at_k"],
                "test_map": test_s["map"], "test_ndcg": test_s["ndcg_at_k"],
                "full_map": task_board.iloc[0]["map"],
            })
    if split_rows:
        split_df = pd.DataFrame(split_rows)
        split_df.to_csv(OUT_DIR / "best_split_evaluation.csv", index=False)
        print(split_df.to_string(index=False))

    print(f"\nAll outputs saved to: {OUT_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Improved AILA retrieval experiments")
    parser.add_argument(
        "--force-embeddings", action="store_true",
        help="Force recompute all embeddings (ignore cache)",
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help="Run cross-encoder reranking (Stage 2). Slower but higher precision.",
    )
    parser.add_argument(
        "--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model for reranking.",
    )
    parser.add_argument("--rerank-k", type=int, default=50, help="Candidates per query for reranking.")
    parser.add_argument("--rerank-batch-size", type=int, default=16, help="Batch size for cross-encoder.")
    return parser.parse_args()


if __name__ == "__main__":
    start = time.time()
    run_improved_experiments(parse_args())
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
