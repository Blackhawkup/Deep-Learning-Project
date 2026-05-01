from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from preprocessing import tokenize


class PassageBM25Retriever:
    def __init__(
        self,
        name: str,
        model_dir: str | Path = "models",
        k1: float = 1.5,
        b: float = 0.75,
        chunk_size: int = 320,
        overlap: int = 80,
        top_passages: int = 3,
        aggregate: str = "max",
    ):
        self.name = name
        self.model_dir = Path(model_dir)
        self.k1 = k1
        self.b = b
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_passages = top_passages
        self.aggregate = aggregate
        self.model = None
        self.doc_ids = None
        self.chunk_doc_ids = None

    def fit(self, documents: pd.DataFrame, text_col: str = "clean_text", id_col: str = "doc_id"):
        doc_ids = documents[id_col].astype(str).tolist()
        tokenized_chunks = []
        chunk_doc_ids = []
        for doc_id, text in zip(doc_ids, documents[text_col].fillna("").astype(str)):
            tokens = tokenize(text)
            if not tokens:
                tokenized_chunks.append([])
                chunk_doc_ids.append(doc_id)
                continue
            step = max(1, self.chunk_size - self.overlap)
            for start in range(0, len(tokens), step):
                chunk = tokens[start : start + self.chunk_size]
                if not chunk:
                    continue
                tokenized_chunks.append(chunk)
                chunk_doc_ids.append(doc_id)
                if start + self.chunk_size >= len(tokens):
                    break

        self.doc_ids = np.asarray(doc_ids)
        self.chunk_doc_ids = np.asarray(chunk_doc_ids)
        self.model = BM25Okapi(tokenized_chunks, k1=self.k1, b=self.b)
        return self

    def save(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "name": self.name,
                "model": self.model,
                "doc_ids": self.doc_ids,
                "chunk_doc_ids": self.chunk_doc_ids,
                "k1": self.k1,
                "b": self.b,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "top_passages": self.top_passages,
                "aggregate": self.aggregate,
            },
            self.model_dir / f"{self.name}_passage_bm25.joblib",
        )

    def load(self):
        payload = joblib.load(self.model_dir / f"{self.name}_passage_bm25.joblib")
        self.name = payload["name"]
        self.model = payload["model"]
        self.doc_ids = payload["doc_ids"]
        self.chunk_doc_ids = payload["chunk_doc_ids"]
        self.k1 = payload.get("k1", self.k1)
        self.b = payload.get("b", self.b)
        self.chunk_size = payload.get("chunk_size", self.chunk_size)
        self.overlap = payload.get("overlap", self.overlap)
        self.top_passages = payload.get("top_passages", self.top_passages)
        self.aggregate = payload.get("aggregate", self.aggregate)
        return self

    def retrieve(self, query_text: str, top_k: int = 10) -> list[dict]:
        self._check_ready()
        chunk_scores = np.asarray(self.model.get_scores(tokenize(query_text or "")), dtype=float)
        doc_scores: dict[str, list[float]] = {}
        positive_idx = np.flatnonzero(chunk_scores > 0)
        if len(positive_idx) == 0:
            positive_idx = np.arange(len(chunk_scores))
        for idx in positive_idx:
            doc_id = str(self.chunk_doc_ids[idx])
            doc_scores.setdefault(doc_id, []).append(float(chunk_scores[idx]))

        rows = []
        for doc_id, scores in doc_scores.items():
            top_scores = sorted(scores, reverse=True)[: self.top_passages]
            if self.aggregate == "sum":
                score = sum(top_scores)
            elif self.aggregate == "mean":
                score = sum(top_scores) / len(top_scores)
            else:
                score = top_scores[0]
            rows.append((doc_id, score))

        rows.sort(key=lambda item: item[1], reverse=True)
        return [
            {"doc_id": doc_id, "rank": rank, "score": float(score)}
            for rank, (doc_id, score) in enumerate(rows[:top_k], start=1)
        ]

    def retrieve_many(
        self,
        queries: pd.DataFrame,
        top_k: int = 10,
        query_col: str = "clean_text",
        query_id_col: str = "query_id",
        task: str = "case",
    ) -> pd.DataFrame:
        rows = []
        for _, query in queries.iterrows():
            for result in self.retrieve(query[query_col], top_k=top_k):
                rows.append(
                    {
                        "query_id": str(query[query_id_col]),
                        "task": task,
                        "model": self.name,
                        **result,
                    }
                )
        return pd.DataFrame(rows)

    def _check_ready(self):
        if self.model is None or self.chunk_doc_ids is None:
            raise RuntimeError("PassageBM25Retriever is not fitted or loaded.")
