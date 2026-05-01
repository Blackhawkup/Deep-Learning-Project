from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


class StatuteClassifierRetriever:
    def __init__(self, name: str = "statute_classifier", model_dir: str | Path = "models"):
        self.name = name
        self.model_dir = Path(model_dir)
        self.vectorizer = None
        self.classifier = None
        self.label_binarizer = None
        self.doc_ids = None

    def fit(
        self,
        queries: pd.DataFrame,
        statute_qrels: pd.DataFrame,
        text_col: str = "clean_text",
        query_id_col: str = "query_id",
    ):
        positives = statute_qrels[statute_qrels["relevance"].astype(int) > 0].copy()
        if positives.empty:
            raise ValueError("No positive statute relevance judgments available for classification.")

        label_map = positives.groupby("query_id")["doc_id"].apply(lambda s: sorted(set(s.astype(str))))
        training = queries[queries[query_id_col].isin(label_map.index)].copy()
        if len(training) < 2:
            raise ValueError("Need at least two labeled queries to train the statute classifier.")

        y_labels = [label_map[qid] for qid in training[query_id_col].astype(str)]
        self.label_binarizer = MultiLabelBinarizer()
        y = self.label_binarizer.fit_transform(y_labels)
        self.doc_ids = np.asarray(self.label_binarizer.classes_, dtype=str)

        self.vectorizer = TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        x = self.vectorizer.fit_transform(training[text_col].fillna("").astype(str))
        self.classifier = OneVsRestClassifier(
            LogisticRegression(max_iter=1000, class_weight="balanced")
        )
        self.classifier.fit(x, y)
        return self

    def save(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "name": self.name,
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
                "label_binarizer": self.label_binarizer,
                "doc_ids": self.doc_ids,
            },
            self.model_dir / f"{self.name}.joblib",
        )

    def load(self):
        payload = joblib.load(self.model_dir / f"{self.name}.joblib")
        self.name = payload["name"]
        self.vectorizer = payload["vectorizer"]
        self.classifier = payload["classifier"]
        self.label_binarizer = payload["label_binarizer"]
        self.doc_ids = payload["doc_ids"]
        return self

    def retrieve(self, query_text: str, top_k: int = 10) -> list[dict]:
        self._check_ready()
        x = self.vectorizer.transform([query_text or ""])
        if hasattr(self.classifier, "predict_proba"):
            scores = self.classifier.predict_proba(x)[0]
        else:
            scores = self.classifier.decision_function(x)[0]
        return self._top_results(np.asarray(scores, dtype=float), top_k)

    def retrieve_many(
        self,
        queries: pd.DataFrame,
        top_k: int = 10,
        query_col: str = "clean_text",
        query_id_col: str = "query_id",
        task: str = "statute",
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
        if self.vectorizer is None or self.classifier is None or self.doc_ids is None:
            raise RuntimeError("StatuteClassifierRetriever is not fitted or loaded.")
