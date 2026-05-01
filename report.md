# AILA Legal Retrieval Pipeline Report

This report outlines the end-to-end workflow of the FIRE 2019 AILA Legal Retrieval project, breaks down what each file is responsible for, highlights the specific Deep Learning components used in the project, and provides a glossary of key Deep Learning terms.

---

## 1. Project Workflow & File Structure

The project follows a standard Information Retrieval (IR) pipeline: Data Gathering $\rightarrow$ Preprocessing $\rightarrow$ Retrieval/Modeling $\rightarrow$ Evaluation $\rightarrow$ Visualization.

### The Detailed Workflow
1. **Setup & Ingestion:** The pipeline starts by using `kagglehub` to download the FIRE 2019 AILA dataset. 
2. **Data Parsing:** Raw text files for queries, cases, and statutes are parsed into structured Pandas DataFrames.
3. **Preprocessing:** Text is normalized (lowercased, special characters removed) and optionally stripped of stopwords to improve keyword matching.
4. **Model Building & Execution:** Multiple retrieval strategies are applied:
   - **Lexical/Sparse models:** TF-IDF and BM25 search for exact keyword overlaps.
   - **Dense/Deep Learning models:** Sentence embeddings retrieve documents based on semantic meaning.
   - **Supervised Classification:** A multi-label classifier is trained specifically to map queries to statutes.
5. **Evaluation:** The retrieved documents are scored against expert judgments (qrels) using rigorous metrics (Precision, Recall, F1, MAP, NDCG).
6. **Advanced Experiments:** Extra scripts fuse multiple models together (combining lexical and semantic searches) and use advanced deep learning techniques like Cross-Encoder reranking.
7. **Dashboard Visualization:** Finally, all results are pulled into an interactive Streamlit web dashboard.

### File Directory Breakdown

| File/Folder | Purpose |
| :--- | :--- |
| **`main.py`** | The primary entry point. Orchestrates downloading data, running the base models, and saving the outputs. |
| **`data_loader.py`** | Handles downloading the dataset from Kaggle and structuring it into `data/queries`, `data/cases`, etc. |
| **`preprocessing.py`** | Contains functions to clean strings and remove stopwords. |
| **`retrieval/tfidf.py`** | Implements the TF-IDF (Term Frequency-Inverse Document Frequency) algorithm for baseline exact-match search. |
| **`retrieval/bm25.py`** | Implements BM25, an advanced and highly robust exact-match keyword search algorithm. |
| **`retrieval/embeddings.py`** | **[Deep Learning]** Uses HuggingFace `sentence-transformers` to embed text into dense vectors for semantic search. |
| **`retrieval/statute_classifier.py`** | Uses supervised machine learning (like Random Forests or Logistic Regression) to map queries to statutes. |
| **`evaluation/metrics.py`** | Calculates IR metrics (Precision@K, Recall@K, MAP, NDCG) comparing model outputs to human relevance judgments. |
| **`dashboard/app.py`** | The frontend Streamlit application that renders graphs and allows interactive querying. |
| **`experiments.py`** | Runs advanced sweeps including stronger Deep Learning models, GPU execution, and Cross-Encoder reranking. |
| **`tune_fusion.py`** | Performs a grid search to find the optimal weights for combining different models (e.g., 50% BM25 + 50% Embeddings). |
| **`models/` & `outputs/`** | Storage directories for cached, trained algorithms and the resulting metric logs. |

---

## 2. The Deep Learning Part

Deep Learning in this pipeline is primarily focused on **Semantic Search**, which solves the "vocabulary mismatch" problem (where a query asks for "car crash" but the legal document says "automobile collision").

1. **Sentence Transformers (Bi-Encoders):** Found in `retrieval/embeddings.py`. The pipeline uses state-of-the-art transformer models like `all-MiniLM-L6-v2` and `all-mpnet-base-v2` (provided by the `sentence-transformers` library). These neural networks read the entire query/document and compress their meaning into a list of numbers (a vector). Documents whose vectors are closest to the query's vector are retrieved.
2. **Cross-Encoder Reranking:** Found in `experiments.py`. For the strongest performance, the system takes the top 100 results from a fast system (like BM25) and passes both the query and the document *simultaneously* through a Deep Learning Cross-Encoder (like `ms-marco-MiniLM-L-6-v2`). The model heavily scrutinizes the exact relationship between words in the query and the document, yielding a highly accurate final score.
3. **GPU Acceleration:** Because Deep Learning involves millions of mathematical operations, `experiments.py` relies on PyTorch configured with CUDA to run these computations blazingly fast on a Graphics Processing Unit (GPU).

---

## 3. Important Deep Learning Key Terms

To fully grasp the advanced mechanics of your pipeline, here are the core Deep Learning and NLP (Natural Language Processing) terms you should know:

> [!NOTE]
> **Embeddings (Dense Vectors)**  
> Think of this as translating text into coordinates on a multi-dimensional map. An embedding is an array of floating-point numbers (e.g., `[0.14, -0.88, 0.31...]`). Words or sentences with similar meanings are plotted physically closer together on this map.

> [!TIP]
> **Transformer Architecture**  
> A breakthrough type of neural network invented by Google in 2017. Transformers process entire sentences at once and use "attention mechanisms" to understand how every word relates to every other word, making them incredibly good at understanding context. ChatGPT and BERT are both built on Transformers.

> [!IMPORTANT]
> **Bi-Encoder**  
> A Deep Learning architecture optimized for speed. It passes the query through the neural network to get a vector, and passes the document through the neural network to get a vector independently. Because documents can be pre-computed and stored, searching millions of records takes milliseconds.

> [!IMPORTANT]
> **Cross-Encoder**  
> A highly accurate but slow Deep Learning architecture. Instead of embedding them separately, it pastes the query and the document together (`[Query] + [Document]`) and runs the entire combined text through the neural network. It's too slow to search a whole database, so it is strictly used as a "Stage 2 Reranker" on a small set of candidates.

> [!NOTE]
> **Cosine Similarity**  
> The mathematical formula used to measure how "close" two embeddings are. It calculates the angle between two vectors. An angle of 0 degrees (Cosine Similarity = 1) means the texts have identical meaning, while a 90-degree angle (Cosine Similarity = 0) means they are completely unrelated.

> [!NOTE]
> **CUDA / Tensor Cores**  
> Software/Hardware technology developed by NVIDIA that allows PyTorch to parallelize the millions of calculations required by Transformers on a GPU, cutting down processing time from hours to seconds.

---

## 4. The Dataset (FIRE 2019 AILA)

The dataset used is the **FIRE 2019 AILA (Artificial Intelligence for Legal Assistance)** dataset. It consists of:
- **Queries:** 50 real-world legal scenarios (fact patterns).
- **Cases:** A corpus of ~3,000 prior Indian Supreme Court precedent cases.
- **Statutes:** A set of ~200 relevant Indian statutes (Acts).
- **Qrels (Relevance Judgments):** Expert annotations indicating which cases and statutes are genuinely relevant to each query.

**What we understood from it:**
The dataset highlights a severe "vocabulary mismatch" problem in legal text. Queries are written as narrative fact patterns (e.g., describing a sequence of events like a landlord-tenant dispute), whereas the precedent cases are dense, formal legal judgments. Simple keyword matching often struggles because the terminology is completely different across narrative facts and formal law, making advanced IR techniques necessary.

---

## 5. Evaluation Metrics & "k"

In Information Retrieval, we don't just care if the right document was found; we care *where* it was ranked. No one looks at page 10 of Google search results.

**What is `k`?**
`k` represents the "cutoff" point for the ranked list. If `k=10`, the system only looks at the top 10 search results to calculate your score. Everything below rank 10 is ignored.

**The Metrics Used:**
- **Precision@K:** Out of the top `K` documents retrieved, what percentage were actually relevant? (Measures noise/accuracy).
- **Recall@K (Ceiling):** Out of ALL the possible relevant documents that exist in the entire database, what percentage did we successfully retrieve in our top `K`? (Measures completeness).
- **F1@K:** The harmonic mean of Precision and Recall. It gives a balanced score so a model doesn't "cheat" by artificially inflating only one metric.
- **MAP (Mean Average Precision):** Evaluates the entire ranked list. It heavily rewards models that put the relevant documents at the very top (e.g., rank 1 or 2) rather than just barely making it into the top 10.
- **NDCG (Normalized Discounted Cumulative Gain):** Similar to MAP, but it accounts for *graded* relevance (e.g., a document being "highly relevant" vs. "somewhat relevant"). It logarithmically discounts the value of a document the further down the ranking it appears.

---

## 6. The Dashboard

The Streamlit dashboard (`dashboard/app.py`) provides an interactive graphical interface to analyze the results without writing code. 

**How it works:**
1. **Metric Loading:** It automatically parses the `outputs/metrics_summary.csv` file generated by the pipeline execution.
2. **Visual Comparisons:** It renders dynamic bar charts allowing you to visually compare `case` and `statute` models across the different metrics (MAP, NDCG, Precision, Recall).
3. **Query Explorer:** It allows you to select any of the 50 dataset queries (or type your own custom scenario) and dynamically run the retrieval models in real-time. It displays the retrieved documents side-by-side with their relevance scores, allowing you to manually inspect *why* a model thought a document was relevant.

---

## 7. Final Scores: Which Model Performed Best and Why?

Based on our final tuned evaluations, the models achieved strong distinct performance markers:

### Top Case Retrieval Model: `case_tfidf`
- **Precision:** 0.40
- **Recall (Ceiling):** 0.38
- **MAP:** 0.425
- **Why it won:** While deep learning embeddings are fantastic for semantic context, TF-IDF performed remarkably well for Indian Legal Cases because case judgments rely heavily on highly specific, rare legal terminology (e.g., Latin maxims, specific act names, exact phrasing of legal tests). TF-IDF statistically identifies and heavily weights these rare terms, making it incredibly precise for exact-match retrieval in dense legal texts.

### Top Statute Retrieval Model: `statute_classifier`
- **Precision:** 0.45
- **Recall (Ceiling):** 0.43
- **MAP:** 0.460
- **Why it won:** Instead of treating statutes as "documents to search through", this model treats statute retrieval as a *Supervised Multi-Label Classification* problem. Because the total number of statutes is relatively small (~200) and we had training data explicitly mapping queries to statutes, training a machine learning classifier to mathematically predict "Statute A" or "Statute B" based on query features easily outperformed traditional search algorithms.
