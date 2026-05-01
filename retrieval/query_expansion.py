"""Pseudo-Relevance Feedback (PRF) query expansion for legal IR."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Legal stop-words: common in judgments but not discriminative
LEGAL_STOPWORDS = frozenset(
    ENGLISH_STOP_WORDS
    | {
        "court", "case", "section", "act", "order", "appellant", "respondent",
        "petitioner", "hon", "shri", "smt", "versus", "judgment", "dated",
        "bench", "appeal", "petition", "filed", "learned", "counsel",
        "submitted", "contended", "argued", "plea", "grounds", "said",
        "matter", "facts", "issues", "held", "decided", "observed",
        "stated", "trial", "civil", "criminal", "writ", "article",
        "india", "indian", "supreme", "high", "district",
    }
)


def _idf_weights(corpus_texts: list[str]) -> dict[str, float]:
    """Compute inverse document frequency for all terms in the corpus."""
    doc_count = len(corpus_texts)
    df: dict[str, int] = {}
    for text in corpus_texts:
        for token in set(re.findall(r"[a-z0-9]+", text.lower())):
            df[token] = df.get(token, 0) + 1
    return {
        token: math.log((doc_count + 1) / (freq + 1)) + 1.0
        for token, freq in df.items()
    }


def extract_expansion_terms(
    feedback_texts: list[str],
    query_tokens: set[str],
    idf: dict[str, float],
    num_terms: int = 20,
    min_len: int = 3,
) -> list[tuple[str, float]]:
    """Extract the most discriminative terms from feedback documents.

    Scores terms by TF-IDF weight, filtering out query terms, stopwords,
    and very short tokens.
    """
    term_scores: dict[str, float] = {}
    for text in feedback_texts:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        tf = Counter(tokens)
        doc_len = max(1, len(tokens))
        for token, count in tf.items():
            if len(token) < min_len:
                continue
            if token in LEGAL_STOPWORDS or token in query_tokens:
                continue
            # Normalized TF * IDF
            score = (count / doc_len) * idf.get(token, 1.0)
            term_scores[token] = term_scores.get(token, 0.0) + score

    ranked = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:num_terms]


class PRFQueryExpander:
    """Pseudo-Relevance Feedback query expansion using BM25 initial retrieval.

    Steps:
    1. Run BM25 on the original query to get top-k feedback documents.
    2. Extract the most informative terms from those documents.
    3. Append expansion terms to the original query (with lower weight via repetition count).
    """

    def __init__(
        self,
        num_feedback_docs: int = 5,
        num_expansion_terms: int = 20,
        expansion_weight: int = 2,  # how many times to repeat expansion terms
        min_term_len: int = 3,
    ):
        self.num_feedback_docs = num_feedback_docs
        self.num_expansion_terms = num_expansion_terms
        self.expansion_weight = expansion_weight
        self.min_term_len = min_term_len
        self.idf: dict[str, float] | None = None

    def build_idf(self, corpus_texts: list[str]) -> "PRFQueryExpander":
        """Pre-compute IDF weights from the document corpus."""
        self.idf = _idf_weights(corpus_texts)
        return self

    def expand_query(
        self,
        query_text: str,
        bm25_retriever,
        documents: pd.DataFrame,
        doc_id_col: str = "doc_id",
        text_col: str = "clean_text",
    ) -> str:
        """Expand a single query using pseudo-relevance feedback."""
        if self.idf is None:
            self.build_idf(documents[text_col].fillna("").astype(str).tolist())

        # Stage 1: get feedback documents
        results = bm25_retriever.retrieve(query_text, top_k=self.num_feedback_docs)
        feedback_doc_ids = {r["doc_id"] for r in results}

        doc_map = documents.set_index(doc_id_col)[text_col].astype(str).to_dict()
        feedback_texts = [
            doc_map[doc_id] for doc_id in feedback_doc_ids if doc_id in doc_map
        ]
        if not feedback_texts:
            return query_text

        # Stage 2: extract expansion terms
        query_tokens = set(re.findall(r"[a-z0-9]+", query_text.lower()))
        expansion = extract_expansion_terms(
            feedback_texts,
            query_tokens,
            self.idf,
            num_terms=self.num_expansion_terms,
            min_len=self.min_term_len,
        )

        # Stage 3: build expanded query
        expansion_str = " ".join(
            f"{term}" for term, _ in expansion
            for _ in range(self.expansion_weight)
        )
        return f"{query_text} {expansion_str}"

    def expand_queries(
        self,
        queries: pd.DataFrame,
        bm25_retriever,
        documents: pd.DataFrame,
        query_col: str = "clean_text",
        doc_id_col: str = "doc_id",
        doc_text_col: str = "clean_text",
    ) -> pd.DataFrame:
        """Expand all queries and return DataFrame with 'expanded_text' column."""
        expanded = queries.copy()
        expanded_texts = []
        for _, row in queries.iterrows():
            expanded_text = self.expand_query(
                row[query_col],
                bm25_retriever,
                documents,
                doc_id_col=doc_id_col,
                text_col=doc_text_col,
            )
            expanded_texts.append(expanded_text)
        expanded["expanded_text"] = expanded_texts
        return expanded
