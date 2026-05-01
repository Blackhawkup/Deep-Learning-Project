from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(
        self,
        name: str,
        model_dir: str | Path = "models",
        max_features: int = 50000,
    ):
        self.name = name
        self.model_dir = Path(model_dir)
        self.max_features = max_features
        self.vectorizer = None
        self.matrix = None
        self.doc_ids = None
        self.documents = None

    def fit(self, documents: pd.DataFrame, text_col: str = "clean_text", id_col: str = "doc_id"):
        self.documents = documents.reset_index(drop=True).copy()
        self.doc_ids = self.documents[id_col].astype(str).to_numpy()
        texts = self.documents[text_col].fillna("").astype(str).tolist()
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            norm="l2",
        )
        self.matrix = self.vectorizer.fit_transform(texts)
        return self

    def save(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "name": self.name,
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
                "doc_ids": self.doc_ids,
            },
            self.model_dir / f"{self.name}_tfidf.joblib",
        )

    def load(self):
        payload = joblib.load(self.model_dir / f"{self.name}_tfidf.joblib")
        self.name = payload["name"]
        self.vectorizer = payload["vectorizer"]
        self.matrix = payload["matrix"]
        self.doc_ids = payload["doc_ids"]
        return self

    def retrieve(self, query_text: str, top_k: int = 10) -> list[dict]:
        self._check_ready()
        query_vec = self.vectorizer.transform([query_text or ""])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        return self._top_results(scores, top_k)

    def retrieve_many(
        self,
        queries: pd.DataFrame,
        top_k: int = 10,
        query_col: str = "clean_text",
        query_id_col: str = "query_id",
        task: str = "case",
    ) -> pd.DataFrame:
        self._check_ready()
        query_matrix = self.vectorizer.transform(queries[query_col].fillna("").astype(str))
        scores = cosine_similarity(query_matrix, self.matrix)
        rows = []
        for row_idx, query_id in enumerate(queries[query_id_col].astype(str)):
            for result in self._top_results(scores[row_idx], top_k):
                rows.append({"query_id": query_id, "task": task, "model": self.name, **result})
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
        if self.vectorizer is None or self.matrix is None or self.doc_ids is None:
            raise RuntimeError("TfidfRetriever is not fitted or loaded.")
