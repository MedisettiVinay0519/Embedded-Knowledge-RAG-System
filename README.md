# Embedded Knowledge RAG System

An Advanced Retrieval-Augmented Generation (RAG) System for:

- Embedded Systems
- RTOS
- Operating Systems
- Communication Protocols
- ARM Cortex
- STM32
- MSP430
- Embedded Security

Built using:

- LangChain
- ChromaDB
- Hybrid Retrieval
- Cross-Encoder Reranking
- Query Rewriting
- Groq LLM
- Streamlit
- RAGAS Evaluation

---

# Features

## Advanced Retrieval Pipeline

- Hybrid Retrieval
  - Dense Vector Search
  - BM25 Retrieval

- Query Rewriting
  - Acronym Expansion
  - Technical Query Optimization

- Cross Encoder Reranking
  - Improves Context Relevance

- Source Grounding
  - Displays Retrieved Sources

---

# Evaluation Dashboard

Includes:

- Faithfulness
- Answer Relevancy
- Context Precision

Visualization:

- Gauge Charts
- Comparison Charts
- Pipeline Improvement Tracking

---

# Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq |
| Embeddings | BAAI/bge-small-en |
| Vector DB | ChromaDB |
| Framework | LangChain |
| Frontend | Streamlit |
| Evaluation | RAGAS |
| Reranking | CrossEncoder |

---

# Project Architecture

```text
User Query
    ↓
Query Rewriting
    ↓
Hybrid Retrieval
    ↓
Cross Encoder Reranking
    ↓
Groq LLM
    ↓
Grounded Response
```

---

# Screenshots

## Main RAG Interface

<img src="screenshots/main_ui.png" width="100%">

---

## Evaluation Dashboard

<img src="screenshots/dashboard.png" width="100%">

---

## Pipeline Comparison

<img src="screenshots/comparison.png" width="100%">

---

# Evaluation Scores

| Metric | Score |
|---|---|
| Faithfulness | 0.8889 |
| Answer Relevancy | 0.9283 |
| Context Precision | 0.9583 |

---

# Folder Structure

```text
Rag_Project/
│
├── app/
│   ├── ingestion/
│   ├── retrieval/
│   ├── reranking/
│   ├── generation/
│   ├── evaluation/
│   └── query_rewrite/
│
├── frontend/
│   ├── streamlit_app.py
│   └── dashboard.py
│
├── data/
├── chroma_db/
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/MedisettiVinay0519/Embedded-Knowledge-RAG-System
cd Rag_Project
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

---

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

---

# Run Streamlit App

```bash
streamlit run frontend/streamlit_app.py
```

---

# Run Dashboard

```bash
streamlit run frontend/dashboard.py
```

---

# Docker Support

Build Docker image:

```bash
docker build -t embedded-rag .
```

Run container:

```bash
docker run -p 8501:8501 embedded-rag
```

---

# Future Improvements

- PDF Highlighting
- Retrieval Score Visualization
- Query Analytics
- Latency Monitoring
- Multi-PDF Knowledge Bases
- Deployment on HuggingFace Spaces
- Authentication

---

# Author

Vinay Medisetti

NITK Surathkal
