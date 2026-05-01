import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing import clean_text
from retrieval import BM25Retriever, EmbeddingRetriever, StatuteClassifierRetriever, TfidfRetriever


DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"


st.set_page_config(page_title="AILA Legal Retrieval", layout="wide")


@st.cache_data(show_spinner=False)
def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


@st.cache_resource(show_spinner=False)
def build_retriever(task: str, model_name: str):
    documents = load_frame(DATA_DIR / ("cases.csv" if task == "case" else "statutes.csv"))
    if documents.empty:
        raise FileNotFoundError("Processed data is missing. Run python main.py first.")
    if "clean_text" not in documents.columns:
        documents["clean_text"] = documents["text"].map(clean_text)

    if model_name.endswith("tfidf"):
        return TfidfRetriever(f"{task}_tfidf", model_dir=MODEL_DIR).fit(documents)
    if model_name.endswith("bm25"):
        return BM25Retriever(f"{task}_bm25", model_dir=MODEL_DIR).fit(documents)
    if model_name.endswith("embeddings"):
        return EmbeddingRetriever(f"{task}_embeddings", model_dir=MODEL_DIR).fit(documents)
    if model_name == "statute_classifier":
        return StatuteClassifierRetriever(model_dir=MODEL_DIR).load()
    raise ValueError(f"Unsupported model: {model_name}")


def plot_metric_bars(metrics: pd.DataFrame, task: str, metric: str):
    subset = metrics[metrics["task"] == task].sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(subset["model"], subset[metric], color="#2f7c8f")
    ax.set_ylim(0, max(1.0, float(subset[metric].max()) + 0.05) if not subset.empty else 1.0)
    ax.set_ylabel(metric)
    ax.set_xlabel("model")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig, clear_figure=True)


def render_results(results: list[dict], documents: pd.DataFrame, task: str):
    docs = documents.set_index("doc_id")
    for result in results:
        doc_id = result["doc_id"]
        if doc_id not in docs.index:
            continue
        doc = docs.loc[doc_id]
        title = doc.get("title", doc_id) if task == "statute" else doc_id
        text = doc.get("text", "")
        with st.expander(f"{result['rank']}. {doc_id} | score {result['score']:.4f}"):
            st.markdown(f"**{title}**")
            st.write(str(text)[:2500])


st.title("AILA Legal Retrieval")

metrics = load_frame(OUTPUT_DIR / "metrics_summary.csv")
queries = load_frame(DATA_DIR / "queries.csv")

if metrics.empty or queries.empty:
    st.warning("Run `python main.py` before opening the dashboard.")
    st.stop()

left, right = st.columns([2, 1])
with right:
    task = st.selectbox("Task", ["case", "statute"])
    metric = st.selectbox("Metric", ["map", "ndcg_at_k", "recall_at_k", "precision_at_k", "f1_at_k"])
    top_k = st.slider("Top K", min_value=1, max_value=50, value=10)

with left:
    st.subheader("Model comparison")
    plot_metric_bars(metrics, task, metric)

st.subheader("Performance insights")
task_metrics = metrics[metrics["task"] == task].copy()
best = task_metrics.sort_values("map", ascending=False).head(1)
if not best.empty:
    row = best.iloc[0]
    st.info(f"Best {task} model by MAP: {row['model']} (MAP {row['map']:.4f})")
st.dataframe(task_metrics.sort_values("map", ascending=False), use_container_width=True)

st.subheader("Query explorer")
available_models = task_metrics["model"].tolist()
model_name = st.selectbox("Model", available_models)
query_mode = st.radio("Query source", ["Dataset query", "Custom query"], horizontal=True)

if query_mode == "Dataset query":
    selected_query = st.selectbox("Query", queries["query_id"].tolist())
    query_text = queries.loc[queries["query_id"] == selected_query, "text"].iloc[0]
    st.text_area("Query text", query_text, height=160, disabled=True)
else:
    query_text = st.text_area("Query text", height=160)

if st.button("Retrieve", type="primary"):
    if not query_text.strip():
        st.warning("Enter a query first.")
        st.stop()
    documents = load_frame(DATA_DIR / ("cases.csv" if task == "case" else "statutes.csv"))
    with st.spinner("Retrieving..."):
        retriever = build_retriever(task, model_name)
        results = retriever.retrieve(clean_text(query_text), top_k=top_k)
    render_results(results, documents, task)
