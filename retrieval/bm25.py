from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from preprocessing import tokenize


class BM25Retriever:
    def __init__(
        self,
        name: str,
        model_dir: str | Path = "models",
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.name = name
        self.model_dir = Path(model_dir)
        self.k1 = k1
        self.b = b
        self.model = None
        self.doc_ids = None
        self.documents = None

    def fit(self, documents: pd.DataFrame, text_col: str = "clean_text", id_col: str = "doc_id"):
        self.documents = documents.reset_index(drop=True).copy()
        self.doc_ids = self.documents[id_col].astype(str).to_numpy()
        tokenized = [tokenize(text) for text in self.documents[text_col].fillna("").astype(str)]
        self.model = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        return self

    def save(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "name": self.name,
                "model": self.model,
                "doc_ids": self.doc_ids,
                "k1": self.k1,
                "b": self.b,
            },
            self.model_dir / f"{self.name}_bm25.joblib",
        )

    def load(self):
        payload = joblib.load(self.model_dir / f"{self.name}_bm25.joblib")
        self.name = payload["name"]
        self.model = payload["model"]
        self.doc_ids = payload["doc_ids"]
        self.k1 = payload.get("k1", self.k1)
        self.b = payload.get("b", self.b)
        return self

    def retrieve(self, query_text: str, top_k: int = 10) -> list[dict]:
        self._check_ready()
        scores = np.asarray(self.model.get_scores(tokenize(query_text or "")), dtype=float)
        return self._top_results(scores, top_k)

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

    def _top_results(self, scores: np.ndarray, top_k: int) -> list[dict]:
        limit = min(top_k, len(scores))
        if limit == 0:
            return []
        top_idx = np.argpartition(-scores, limit - 1)[:limit]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [
            {
                "doc_id": str(self.doc_ids[idx]),
                "rank": rank,
                "score": float(scores[idx]),
            }
            for rank, idx in enumerate(top_idx, start=1)
        ]

    def _check_ready(self):
        if self.model is None or self.doc_ids is None:
            raise RuntimeError("BM25Retriever is not fitted or loaded.")
