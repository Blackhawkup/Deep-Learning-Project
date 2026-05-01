from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd


class EmbeddingRetriever:
    """Dense retriever using sentence-transformer models.

    Improvements over the original:
    - Multiple pooling strategies for long documents (mean, max, weighted_mean)
    - Instruction/prefix-based model support (E5, BGE)
    - Better defaults for legal IR: larger max_chars, sliding window enabled
    - Asymmetric encoding: query vs document can use different prefixes
    """

    POOL_STRATEGIES = ("mean", "max", "weighted_mean")

    def __init__(
        self,
        name: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_dir: str | Path = "models",
        batch_size: int = 32,
        max_chars: int = 8000,
        window_tokens: int | None = None,
        window_stride: int = 128,
        pool_strategy: str = "mean",
        query_prefix: str = "",
        document_prefix: str = "",
        preprocessing_signature: str = "default",
    ):
        self.name = name
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.window_tokens = window_tokens
        self.window_stride = window_stride
        self.pool_strategy = pool_strategy
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.preprocessing_signature = preprocessing_signature
        self.model = None
        self.embeddings = None
        self.doc_ids = None
        self.documents = None
        self.cache_key = None

    def fit(
        self,
        documents: pd.DataFrame,
        text_col: str = "clean_text",
        id_col: str = "doc_id",
        force: bool = False,
    ):
        self.documents = documents.reset_index(drop=True).copy()
        self.doc_ids = self.documents[id_col].astype(str).to_numpy()
        cache_key = self._cache_key(self.documents, text_col=text_col, id_col=id_col)

        cache_path = self.model_dir / f"{self.name}_embeddings.npy"
        id_path = self.model_dir / f"{self.name}_embedding_ids.joblib"
        key_path = self.model_dir / f"{self.name}_embedding_key.json"
        if cache_path.exists() and id_path.exists() and key_path.exists() and not force:
            cached_ids = joblib.load(id_path)
            cached_key = json.loads(key_path.read_text(encoding="utf-8")).get("cache_key")
            if list(cached_ids) == list(self.doc_ids) and cached_key == cache_key:
                self.embeddings = np.load(cache_path)
                self.cache_key = cache_key
                self._load_sentence_model()
                return self

        self._load_sentence_model()
        texts = self.documents[text_col].fillna("").astype(str).tolist()
        # Documents get the document prefix
        prefixed = [f"{self.document_prefix}{t}" for t in texts]
        self.embeddings = self._encode_texts(prefixed, show_progress_bar=True)
        self.cache_key = cache_key
        self.model_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, self.embeddings)
        joblib.dump(self.doc_ids, id_path)
        key_path.write_text(json.dumps({"cache_key": cache_key}, indent=2), encoding="utf-8")
        self.save_metadata()
        return self

    def save_metadata(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "name": self.name,
                "model_name": self.model_name,
                "doc_ids": self.doc_ids,
                "max_chars": self.max_chars,
                "window_tokens": self.window_tokens,
                "window_stride": self.window_stride,
                "pool_strategy": self.pool_strategy,
                "query_prefix": self.query_prefix,
                "document_prefix": self.document_prefix,
                "preprocessing_signature": self.preprocessing_signature,
                "cache_key": self.cache_key,
            },
            self.model_dir / f"{self.name}_embedding_meta.joblib",
        )

    def load(self):
        payload = joblib.load(self.model_dir / f"{self.name}_embedding_meta.joblib")
        self.name = payload["name"]
        self.model_name = payload["model_name"]
        self.doc_ids = payload["doc_ids"]
        self.max_chars = payload.get("max_chars", self.max_chars)
        self.window_tokens = payload.get("window_tokens", self.window_tokens)
        self.window_stride = payload.get("window_stride", self.window_stride)
        self.pool_strategy = payload.get("pool_strategy", self.pool_strategy)
        self.query_prefix = payload.get("query_prefix", self.query_prefix)
        self.document_prefix = payload.get("document_prefix", self.document_prefix)
        self.preprocessing_signature = payload.get(
            "preprocessing_signature", self.preprocessing_signature
        )
        self.cache_key = payload.get("cache_key")
        self.embeddings = np.load(self.model_dir / f"{self.name}_embeddings.npy")
        self._load_sentence_model()
        return self

    def retrieve(self, query_text: str, top_k: int = 10) -> list[dict]:
        self._check_ready()
        prefixed = f"{self.query_prefix}{query_text or ''}"
        query_embedding = self._encode_texts([prefixed], show_progress_bar=False)[0]
        scores = np.asarray(np.dot(self.embeddings, query_embedding), dtype=float)
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
        texts = queries[query_col].fillna("").astype(str).tolist()
        prefixed = [f"{self.query_prefix}{t}" for t in texts]
        query_embeddings = self._encode_texts(prefixed, show_progress_bar=True)
        scores = np.matmul(query_embeddings, self.embeddings.T)
        rows = []
        for row_idx, query_id in enumerate(queries[query_id_col].astype(str)):
            for result in self._top_results(scores[row_idx], top_k):
                rows.append({"query_id": query_id, "task": task, "model": self.name, **result})
        return pd.DataFrame(rows)

    def _load_sentence_model(self):
        if self.model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install -r requirements.txt"
            ) from exc
        self.model = SentenceTransformer(self.model_name)

    def _cache_key(self, documents: pd.DataFrame, text_col: str, id_col: str) -> str:
        frame = documents[[id_col, text_col]].copy()
        frame[id_col] = frame[id_col].astype(str)
        frame[text_col] = frame[text_col].fillna("").astype(str)
        sorted_frame = frame.sort_values(id_col)
        payload = {
            "model_name": self.model_name,
            "max_chars": self.max_chars,
            "window_tokens": self.window_tokens,
            "window_stride": self.window_stride,
            "pool_strategy": self.pool_strategy,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "preprocessing_signature": self.preprocessing_signature,
            "text_col": text_col,
            "doc_ids_ordered": frame[id_col].tolist(),
            "doc_ids_sorted": sorted_frame[id_col].tolist(),
            "text_sha256": hashlib.sha256(
                "\n".join(
                    f"{row[id_col]}\t{row[text_col]}" for _, row in sorted_frame.iterrows()
                ).encode("utf-8")
            ).hexdigest(),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _encode_texts(self, texts: list[str], show_progress_bar: bool) -> np.ndarray:
        if not self.window_tokens or self.window_tokens <= 0:
            return self.model.encode(
                [self._truncate(text) for text in texts],
                batch_size=self.batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=True,
            )

        vectors = []
        for text in texts:
            chunks = self._token_windows(text)
            chunk_embeddings = self.model.encode(
                chunks,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            chunk_embeddings = np.asarray(chunk_embeddings, dtype=float)

            if self.pool_strategy == "max":
                # Max pooling: take element-wise max across chunks
                vector = np.max(chunk_embeddings, axis=0)
            elif self.pool_strategy == "weighted_mean":
                # Weighted mean: first and last chunks get 1.5x weight
                # (legal docs often state issues at start, ratio at end)
                n_chunks = len(chunk_embeddings)
                weights = np.ones(n_chunks)
                if n_chunks >= 3:
                    weights[0] = 1.5   # first chunk (case intro / issues)
                    weights[-1] = 1.5  # last chunk (ratio / order)
                weights = weights / weights.sum()
                vector = np.average(chunk_embeddings, axis=0, weights=weights)
            else:
                # Default: mean pooling
                vector = np.mean(chunk_embeddings, axis=0)

            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            vectors.append(vector)
        return np.asarray(vectors, dtype=float)

    def _token_windows(self, text: str) -> list[str]:
        tokens = self._truncate(text).split()
        if not tokens:
            return [""]
        if self.window_tokens and len(tokens) <= self.window_tokens:
            return [" ".join(tokens)]

        stride = max(1, self.window_stride)
        chunks = []
        for start in range(0, len(tokens), stride):
            chunk = tokens[start : start + self.window_tokens]
            if not chunk:
                continue
            chunks.append(" ".join(chunk))
            if start + self.window_tokens >= len(tokens):
                break
        return chunks

    def _truncate(self, text: str) -> str:
        text = str(text or "")
        if self.max_chars <= 0 or len(text) <= self.max_chars:
            return text
        return text[: self.max_chars]

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
        if self.model is None or self.embeddings is None or self.doc_ids is None:
            raise RuntimeError("EmbeddingRetriever is not fitted or loaded.")
