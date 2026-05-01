import os
from pathlib import Path

import numpy as np
import pandas as pd


class CrossEncoderReranker:
    """Cross-encoder reranker with passage-level scoring support.

    Improvements:
    - Passage-level reranking: splits long documents into overlapping chunks,
      scores each chunk against the query, and takes the max score.
      This is critical for legal documents where the relevant passage
      may be buried in thousands of words.
    - Configurable max_length for models with different context windows.
    """

    def __init__(
        self,
        name: str,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        model_dir: str | Path = "models",
        max_length: int = 512,
        batch_size: int = 12,
        max_doc_chars: int = 7000,
        passage_rerank: bool = False,
        passage_size: int = 400,  # words per passage
        passage_stride: int = 200,  # overlap
    ):
        self.name = name
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.batch_size = batch_size
        self.max_doc_chars = max_doc_chars
        self.passage_rerank = passage_rerank
        self.passage_size = passage_size
        self.passage_stride = passage_stride
        self.model = None

    def rerank(
        self,
        candidate_run: pd.DataFrame,
        queries: pd.DataFrame,
        documents: pd.DataFrame,
        task: str,
        candidate_k: int = 50,
        output_k: int = 10,
    ) -> pd.DataFrame:
        self._load_model()
        query_map = queries.set_index("query_id")["text"].astype(str).to_dict()
        doc_map = documents.set_index("doc_id")["text"].astype(str).to_dict()

        rows = []
        for query_id, group in candidate_run.groupby("query_id"):
            group = group.sort_values("rank").head(candidate_k)
            query_text = query_map.get(str(query_id), "")

            if self.passage_rerank:
                scores = self._passage_score_batch(
                    query_text,
                    [doc_map.get(str(doc_id), "") for doc_id in group["doc_id"].astype(str)],
                )
            else:
                pairs = [
                    [query_text, doc_map.get(str(doc_id), "")[: self.max_doc_chars]]
                    for doc_id in group["doc_id"].astype(str)
                ]
                scores = self.model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                )

            ranked = group[["query_id", "task", "doc_id"]].copy()
            ranked["score"] = np.asarray(scores, dtype=float)
            ranked = ranked.sort_values("score", ascending=False).head(output_k).copy()
            ranked["rank"] = range(1, len(ranked) + 1)
            rows.append(ranked)

        result = pd.concat(rows, ignore_index=True)
        result["task"] = task
        result["model"] = self.name
        return result[["query_id", "task", "model", "doc_id", "rank", "score"]]

    def _passage_score_batch(
        self,
        query_text: str,
        doc_texts: list[str],
    ) -> list[float]:
        """Score each document by its best-matching passage.

        For each document, split into overlapping passages, score each
        passage against the query, and return the maximum score.
        """
        all_pairs = []
        doc_passage_counts = []

        for doc_text in doc_texts:
            passages = self._split_passages(doc_text)
            doc_passage_counts.append(len(passages))
            for passage in passages:
                all_pairs.append([query_text, passage])

        if not all_pairs:
            return [0.0] * len(doc_texts)

        all_scores = self.model.predict(
            all_pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Aggregate: take max score per document
        doc_scores = []
        idx = 0
        for count in doc_passage_counts:
            if count == 0:
                doc_scores.append(0.0)
            else:
                passage_scores = all_scores[idx : idx + count]
                doc_scores.append(float(np.max(passage_scores)))
                idx += count

        return doc_scores

    def _split_passages(self, text: str) -> list[str]:
        """Split a document into overlapping passages for passage-level scoring."""
        words = text.split()
        if len(words) <= self.passage_size:
            return [text[: self.max_doc_chars]]

        passages = []
        stride = max(1, self.passage_stride)
        for start in range(0, len(words), stride):
            chunk = words[start : start + self.passage_size]
            if not chunk:
                continue
            passages.append(" ".join(chunk))
            if start + self.passage_size >= len(words):
                break

        return passages

    def _load_model(self):
        if self.model is not None:
            return
        os.environ.setdefault("TRANSFORMERS_CACHE", str(self.model_dir / "hf_cache"))
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers and torch are required for cross-encoder reranking."
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(self.model_name, device=device, max_length=self.max_length)
