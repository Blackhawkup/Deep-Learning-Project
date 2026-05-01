"""Hybrid retrieval combining multiple retrieval signals with proper normalization."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


NormStrategy = Literal["minmax", "zscore", "rank"]


def _normalize_scores(
    scores: pd.Series,
    strategy: NormStrategy = "zscore",
) -> pd.Series:
    """Normalize scores within a single query's result set."""
    if len(scores) <= 1:
        return pd.Series(np.ones(len(scores)), index=scores.index)

    if strategy == "minmax":
        low, high = scores.min(), scores.max()
        if high == low:
            return pd.Series(np.ones(len(scores)), index=scores.index)
        return (scores - low) / (high - low)

    elif strategy == "zscore":
        mean, std = scores.mean(), scores.std()
        if std == 0:
            return pd.Series(np.zeros(len(scores)), index=scores.index)
        return (scores - mean) / std

    elif strategy == "rank":
        # Rank-based normalization: rank / N
        n = len(scores)
        ranks = scores.rank(ascending=False)
        return 1.0 - (ranks - 1) / max(1, n - 1)

    raise ValueError(f"Unknown normalization strategy: {strategy}")


def normalize_run(
    run: pd.DataFrame,
    score_name: str,
    strategy: NormStrategy = "zscore",
) -> pd.DataFrame:
    """Normalize scores per query in a retrieval run."""
    frame = run[["query_id", "doc_id", "score"]].copy()
    frame[score_name] = frame.groupby("query_id")["score"].transform(
        lambda s: _normalize_scores(s, strategy)
    )
    return frame[["query_id", "doc_id", score_name]]


def weighted_rrf(
    runs: dict[str, pd.DataFrame],
    weights: dict[str, float],
    model_name: str,
    task: str,
    top_k: int,
    k_const: int = 60,
) -> pd.DataFrame:
    """Weighted Reciprocal Rank Fusion.

    Each system contributes 1/(k + rank) weighted by its system weight.
    """
    scores: dict[tuple[str, str], float] = {}
    for name, run in runs.items():
        w = weights.get(name, 1.0)
        for row in run.itertuples(index=False):
            key = (str(row.query_id), str(row.doc_id))
            scores[key] = scores.get(key, 0.0) + w / (k_const + int(row.rank))

    frame = pd.DataFrame(
        [(qid, did, s) for (qid, did), s in scores.items()],
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


def fuse_runs_normalized(
    runs: dict[str, pd.DataFrame],
    weights: dict[str, float],
    model_name: str,
    task: str,
    top_k: int,
    norm_strategy: NormStrategy = "zscore",
) -> pd.DataFrame:
    """Fuse multiple retrieval runs using normalized score combination.

    Unlike simple min-max, this supports z-score and rank-based normalization
    which are more robust to score distribution differences between systems.
    """
    merged = None
    for name, run in runs.items():
        current = normalize_run(run, name, strategy=norm_strategy)
        if merged is None:
            merged = current
        else:
            merged = merged.merge(current, on=["query_id", "doc_id"], how="outer")

    if merged is None:
        return pd.DataFrame(columns=["query_id", "task", "model", "doc_id", "rank", "score"])

    merged = merged.fillna(0.0)
    merged["score"] = sum(
        weights.get(name, 0.0) * merged[name]
        for name in runs
        if name in merged.columns
    )

    ranked = []
    for query_id, group in merged.groupby("query_id"):
        group = group.sort_values("score", ascending=False).head(top_k).copy()
        group["rank"] = range(1, len(group) + 1)
        group["model"] = model_name
        group["task"] = task
        ranked.append(group[["query_id", "task", "model", "doc_id", "rank", "score"]])
    return pd.concat(ranked, ignore_index=True)


class HybridRetriever:
    """Orchestrates multiple retrievers with configurable fusion.

    Supports:
    - Multiple retrieval systems (BM25, TF-IDF, embeddings, classifier)
    - Multiple fusion strategies (weighted score fusion, weighted RRF)
    - Multiple normalization methods (min-max, z-score, rank-based)
    - Grid search over fusion weights
    """

    def __init__(
        self,
        retrievers: dict[str, object],
        candidate_k: int = 100,
        output_k: int = 10,
        fusion: Literal["score", "rrf"] = "rrf",
        norm_strategy: NormStrategy = "zscore",
        weights: dict[str, float] | None = None,
        rrf_k: int = 60,
    ):
        self.retrievers = retrievers
        self.candidate_k = candidate_k
        self.output_k = output_k
        self.fusion = fusion
        self.norm_strategy = norm_strategy
        self.weights = weights or {name: 1.0 for name in retrievers}
        self.rrf_k = rrf_k

    def retrieve(
        self,
        queries: pd.DataFrame,
        task: str,
        query_col: str = "clean_text",
    ) -> pd.DataFrame:
        """Run all retrievers and fuse results."""
        runs = {}
        for name, retriever in self.retrievers.items():
            run = retriever.retrieve_many(
                queries, top_k=self.candidate_k, task=task, query_col=query_col
            )
            runs[name] = run

        model_name = f"{task}_hybrid_{'_'.join(sorted(runs.keys()))}"

        if self.fusion == "rrf":
            return weighted_rrf(
                runs, self.weights, model_name, task,
                self.output_k, k_const=self.rrf_k,
            )
        else:
            return fuse_runs_normalized(
                runs, self.weights, model_name, task,
                self.output_k, norm_strategy=self.norm_strategy,
            )

    def grid_search_weights(
        self,
        queries: pd.DataFrame,
        qrels: pd.DataFrame,
        task: str,
        evaluate_fn,
        step: float = 0.1,
        query_col: str = "clean_text",
    ) -> tuple[dict[str, float], dict, pd.DataFrame]:
        """Grid search over fusion weights, returning best weights and metrics."""
        from itertools import product as itertools_product

        # First, generate all candidate runs
        runs = {}
        for name, retriever in self.retrievers.items():
            runs[name] = retriever.retrieve_many(
                queries, top_k=self.candidate_k, task=task, query_col=query_col
            )

        names = list(runs.keys())
        slots = int(round(1.0 / step))
        results = []
        best_weights = None
        best_metrics = None

        for raw in itertools_product(range(slots + 1), repeat=len(names)):
            if sum(raw) != slots:
                continue
            weights = dict(zip(names, [v / slots for v in raw]))
            if sum(1 for v in weights.values() if v > 0) < 2 and len(names) > 1:
                continue

            model_name = f"{task}_grid"
            if self.fusion == "rrf":
                fused = weighted_rrf(runs, weights, model_name, task, self.output_k, self.rrf_k)
            else:
                fused = fuse_runs_normalized(runs, weights, model_name, task, self.output_k, self.norm_strategy)

            _, summary = evaluate_fn(fused, qrels, k=self.output_k)
            results.append({**summary, "weights": str(weights)})

            if best_metrics is None or summary["map"] > best_metrics["map"]:
                best_metrics = summary
                best_weights = weights

        return best_weights, best_metrics, pd.DataFrame(results)
