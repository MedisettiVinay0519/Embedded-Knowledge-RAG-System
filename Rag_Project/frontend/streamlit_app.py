import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import sys
from pathlib import Path

# =========================================================
# ROOT PATH SETUP
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT_DIR))

# =========================================================
# IMPORTS
# =========================================================

import streamlit as st

from app.generation.rag_chain import generate_response

from app.ingestion.loader import load_documents
from app.ingestion.chunker import split_documents
from app.ingestion.vectordb import create_vector_store

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Embedded Knowledge RAG System",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🤖 Embedded Knowledge RAG System")

st.markdown("""
Advanced RAG System for:

- Embedded Systems
- RTOS
- Communication Protocols
- ARM Cortex
- Operating Systems

### Features

- Hybrid Retrieval
- Query Rewriting
- Reranking
- Groq LLM
- ChromaDB
- RAG Evaluation
- Source Grounding
""")

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ System Pipeline")

    st.markdown("""
    1. Query Rewriting  
    2. Hybrid Retrieval  
    3. Reranking  
    4. Groq LLM Generation  
    5. Source Grounding  
    """)

    st.header("📊 Evaluation Scores")

    st.metric("Faithfulness", "0.8889")
    st.metric("Answer Relevancy", "0.9283")
    st.metric("Context Precision", "0.9583")

# =========================================================
# PDF UPLOAD
# =========================================================

st.subheader("📄 Upload Embedded Systems / OS PDF")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    save_dir = ROOT_DIR / "data"

    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / uploaded_file.name

    with open(save_path, "wb") as f:

        f.write(uploaded_file.getbuffer())

    st.success(f"{uploaded_file.name} uploaded successfully.")

    # =====================================================
    # VECTOR DB REBUILD
    # =====================================================

    if st.button("🔄 Rebuild Vector Database"):

        with st.spinner("Rebuilding vector database..."):

            # =================================================
            # LOAD DOCUMENTS
            # =================================================

            docs = load_documents(str(ROOT_DIR / "data"))

            # =================================================
            # SPLIT DOCUMENTS
            # =================================================

            chunks = split_documents(docs)

            # =================================================
            # CREATE VECTOR STORE
            # =================================================

            create_vector_store(chunks)

        st.success("✅ Vector database rebuilt successfully.")

# =========================================================
# USER INPUT
# =========================================================

st.subheader("💬 Ask Questions")

query = st.text_input(
    "Enter your question:",
    placeholder="Example: Explain SPI communication"
)

# =========================================================
# GENERATE RESPONSE
# =========================================================

if st.button("Generate Answer"):

    if query.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Generating response..."):

            result = generate_response(query)

        # =================================================
        # REWRITTEN QUERY
        # =================================================

        st.subheader("🔄 Rewritten Query")

        st.info(result["rewritten_query"])

        # =================================================
        # FINAL ANSWER
        # =================================================

        st.subheader("🧠 Generated Answer")

        st.write(result["answer"])

        # =================================================
        # LATENCY METRICS
        # =================================================

        st.subheader("⏱️ Pipeline Latency")

        latency = result["latency"]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Rewrite",
            f"{latency['rewrite_time']:.2f}s"
        )

        col2.metric(
            "Retrieval",
            f"{latency['retrieval_time']:.2f}s"
        )

        col3.metric(
            "Reranking",
            f"{latency['rerank_time']:.2f}s"
        )

        col4.metric(
            "Generation",
            f"{latency['generation_time']:.2f}s"
        )

        st.metric(
            "Total Pipeline Time",
            f"{latency['total_time']:.2f}s"
        )

        # =================================================
        # SOURCES
        # =================================================

        st.subheader("📚 Retrieved Sources")

        for item in result["source_data"]:

            st.markdown(
                f"""
                ### 📄 {item['source']}

                - **Page:** {item['page']}
                - **Rerank Score:** {item['score']}
                """
            )

            with st.expander("View Retrieved Chunk"):

                st.write(item["content"])

        # =================================================
        # ALL CONTEXTS
        # =================================================

        with st.expander("📄 All Retrieved Contexts"):

            for i, context in enumerate(result["contexts"]):

                st.markdown(f"### Chunk {i+1}")

                st.write(context)

                st.markdown("---")