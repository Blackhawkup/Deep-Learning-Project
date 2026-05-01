"""Two-stage retrieval pipeline: recall-oriented first stage + precision reranking.

Stage 1: Cast a wide net with hybrid retrieval (BM25 + TF-IDF + embeddings + query expansion)
Stage 2: Re-rank top candidates with cross-encoder (optionally passage-level)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from retrieval.bm25 import BM25Retriever
from retrieval.cross_encoder import CrossEncoderReranker
from retrieval.embeddings import EmbeddingRetriever
from retrieval.hybrid import fuse_runs_normalized, weighted_rrf
from retrieval.query_expansion import PRFQueryExpander
from retrieval.tfidf import TfidfRetriever


class TwoStagePipeline:
    """Two-stage legal IR pipeline.

    Stage 1 maximizes recall by combining:
    - BM25 (lexical, handles legal terminology well)
    - TF-IDF (complementary lexical signal with bigrams)
    - Embedding retriever (semantic understanding)
    - Optional query expansion (bridges vocabulary gap)

    Stage 2 maximizes precision by:
    - Cross-encoder reranking with passage-level scoring
    """

    def __init__(
        self,
        model_dir: str | Path = "models",
        # Stage 1 config
        stage1_k: int = 100,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        embedding_max_chars: int = 10000,
        embedding_window_tokens: int = 256,
        embedding_window_stride: int = 128,
        embedding_pool: str = "max",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        use_query_expansion: bool = True,
        prf_feedback_docs: int = 5,
        prf_expansion_terms: int = 15,
        prf_expansion_weight: int = 2,
        # Fusion config
        fusion_method: str = "rrf",
        fusion_weights: dict[str, float] | None = None,
        rrf_k: int = 60,
        # Stage 2 config
        use_reranker: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_k: int = 50,
        reranker_passage: bool = True,
        reranker_passage_size: int = 350,
        reranker_passage_stride: int = 175,
        reranker_batch_size: int = 16,
        # Output config
        output_k: int = 10,
    ):
        self.model_dir = Path(model_dir)
        self.stage1_k = stage1_k
        self.embedding_model = embedding_model
        self.embedding_max_chars = embedding_max_chars
        self.embedding_window_tokens = embedding_window_tokens
        self.embedding_window_stride = embedding_window_stride
        self.embedding_pool = embedding_pool
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.use_query_expansion = use_query_expansion
        self.prf_feedback_docs = prf_feedback_docs
        self.prf_expansion_terms = prf_expansion_terms
        self.prf_expansion_weight = prf_expansion_weight
        self.fusion_method = fusion_method
        self.fusion_weights = fusion_weights
        self.rrf_k = rrf_k
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.reranker_k = reranker_k
        self.reranker_passage = reranker_passage
        self.reranker_passage_size = reranker_passage_size
        self.reranker_passage_stride = reranker_passage_stride
        self.reranker_batch_size = reranker_batch_size
        self.output_k = output_k

        # Built during fit
        self._bm25 = None
        self._tfidf = None
        self._embedding = None
        self._expander = None
        self._reranker = None
        self._documents = None

    def fit(
        self,
        documents: pd.DataFrame,
        task: str = "case",
        text_col: str = "clean_text",
        id_col: str = "doc_id",
        force_embeddings: bool = False,
    ) -> "TwoStagePipeline":
        """Fit all stage-1 retrievers on the document collection."""
        self._documents = documents
        self._task = task

        print(f"[Pipeline] Fitting BM25 for {task}...")
        self._bm25 = BM25Retriever(
            f"{task}_pipe_bm25",
            model_dir=self.model_dir,
            k1=self.bm25_k1,
            b=self.bm25_b,
        ).fit(documents, text_col=text_col, id_col=id_col)

        print(f"[Pipeline] Fitting TF-IDF for {task}...")
        self._tfidf = TfidfRetriever(
            f"{task}_pipe_tfidf",
            model_dir=self.model_dir,
        ).fit(documents, text_col=text_col, id_col=id_col)

        print(f"[Pipeline] Fitting embedding retriever for {task} ({self.embedding_model})...")
        self._embedding = EmbeddingRetriever(
            f"{task}_pipe_emb",
            model_name=self.embedding_model,
            model_dir=self.model_dir,
            batch_size=16,
            max_chars=self.embedding_max_chars,
            window_tokens=self.embedding_window_tokens,
            window_stride=self.embedding_window_stride,
            pool_strategy=self.embedding_pool,
            preprocessing_signature=f"pipe_{task}_v1",
        ).fit(documents, text_col=text_col, id_col=id_col, force=force_embeddings)

        if self.use_query_expansion:
            print(f"[Pipeline] Building IDF for query expansion...")
            self._expander = PRFQueryExpander(
                num_feedback_docs=self.prf_feedback_docs,
                num_expansion_terms=self.prf_expansion_terms,
                expansion_weight=self.prf_expansion_weight,
            ).build_idf(documents[text_col].fillna("").astype(str).tolist())

        if self.use_reranker:
            print(f"[Pipeline] Preparing cross-encoder ({self.reranker_model})...")
            self._reranker = CrossEncoderReranker(
                f"{task}_pipe_rerank",
                model_name=self.reranker_model,
                model_dir=self.model_dir,
                batch_size=self.reranker_batch_size,
                passage_rerank=self.reranker_passage,
                passage_size=self.reranker_passage_size,
                passage_stride=self.reranker_passage_stride,
            )

        return self

    def retrieve(
        self,
        queries: pd.DataFrame,
        query_col: str = "clean_text",
        query_id_col: str = "query_id",
    ) -> pd.DataFrame:
        """Run the full two-stage pipeline."""
        task = self._task

        # ----- Stage 1: Candidate retrieval -----
        print(f"[Pipeline] Stage 1: Retrieving candidates (top-{self.stage1_k})...")
        runs = {}

        # BM25
        run_bm25 = self._bm25.retrieve_many(
            queries, top_k=self.stage1_k, query_col=query_col, task=task
        )
        runs["bm25"] = run_bm25

        # TF-IDF
        run_tfidf = self._tfidf.retrieve_many(
            queries, top_k=self.stage1_k, query_col=query_col, task=task
        )
        runs["tfidf"] = run_tfidf

        # Embeddings
        run_emb = self._embedding.retrieve_many(
            queries, top_k=self.stage1_k, query_col=query_col, task=task
        )
        runs["emb"] = run_emb

        # Query expansion + BM25 (additional signal)
        if self._expander is not None:
            print(f"[Pipeline] Running query expansion...")
            expanded = self._expander.expand_queries(
                queries, self._bm25, self._documents,
                query_col=query_col,
                doc_text_col="clean_text",
            )
            run_expanded = self._bm25.retrieve_many(
                expanded, top_k=self.stage1_k,
                query_col="expanded_text", task=task,
            )
            run_expanded["model"] = f"{task}_pipe_bm25_expanded"
            runs["bm25_expanded"] = run_expanded

        # Fuse
        print(f"[Pipeline] Fusing {len(runs)} retrieval signals...")
        weights = self.fusion_weights or {name: 1.0 for name in runs}
        model_name = f"{task}_pipeline_stage1"

        if self.fusion_method == "rrf":
            stage1_run = weighted_rrf(
                runs, weights, model_name, task,
                top_k=self.stage1_k, k_const=self.rrf_k,
            )
        else:
            stage1_run = fuse_runs_normalized(
                runs, weights, model_name, task,
                top_k=self.stage1_k, norm_strategy="zscore",
            )

        # ----- Stage 2: Reranking -----
        if self._reranker is not None:
            print(f"[Pipeline] Stage 2: Cross-encoder reranking (top-{self.reranker_k})...")
            final_run = self._reranker.rerank(
                stage1_run,
                queries,
                self._documents,
                task=task,
                candidate_k=self.reranker_k,
                output_k=self.output_k,
            )
            final_run["model"] = f"{task}_pipeline_full"
        else:
            final_run = stage1_run.sort_values(["query_id", "rank"]).groupby("query_id").head(self.output_k).copy()
            final_run["model"] = f"{task}_pipeline_stage1_only"
            # Recompute ranks after head
            ranked = []
            for _, group in final_run.groupby("query_id"):
                group = group.sort_values("score", ascending=False).copy()
                group["rank"] = range(1, len(group) + 1)
                ranked.append(group)
            final_run = pd.concat(ranked, ignore_index=True)

        return final_run

    def get_stage1_runs(self) -> dict:
        """Return individual stage-1 runs for analysis."""
        return getattr(self, "_last_runs", {})
