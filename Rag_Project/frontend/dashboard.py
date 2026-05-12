import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("📊 RAG Evaluation Dashboard")

st.markdown("""
Evaluation and benchmarking dashboard for the:

### Embedded Knowledge RAG System

This dashboard visualizes:
- RAGAS evaluation metrics
- Pipeline improvements
- Retrieval optimization impact
""")

# =========================================================
# FINAL EVALUATION SCORES
# =========================================================

faithfulness = 0.8889
answer_relevancy = 0.9283
context_precision = 0.9583

# =========================================================
# METRIC CARDS
# =========================================================

st.subheader("🎯 Final Evaluation Metrics")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Faithfulness",
    f"{faithfulness:.4f}"
)

col2.metric(
    "Answer Relevancy",
    f"{answer_relevancy:.4f}"
)

col3.metric(
    "Context Precision",
    f"{context_precision:.4f}"
)

# =========================================================
# GAUGE CHARTS
# =========================================================

st.subheader("📈 Metric Visualization")

gauge1, gauge2, gauge3 = st.columns(3)

# =========================================================
# FAITHFULNESS
# =========================================================

with gauge1:

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=faithfulness,
        title={'text': "Faithfulness"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "green"}
        }
    ))

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# ANSWER RELEVANCY
# =========================================================

with gauge2:

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=answer_relevancy,
        title={'text': "Answer Relevancy"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "blue"}
        }
    ))

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# CONTEXT PRECISION
# =========================================================

with gauge3:

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=context_precision,
        title={'text': "Context Precision"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "orange"}
        }
    ))

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# PIPELINE COMPARISON
# =========================================================

st.subheader("⚙️ Evaluation Comparison Dashboard")

comparison_df = pd.DataFrame({

    "Pipeline": [

        "Basic RAG",

        "RAG + Reranking",

        "RAG + Hybrid Retrieval",

        "RAG + Query Rewriting",

        "Final Advanced RAG"
    ],

    "Faithfulness": [

        0.72,

        0.79,

        0.84,

        0.87,

        0.8889
    ],

    "Answer Relevancy": [

        0.81,

        0.86,

        0.89,

        0.91,

        0.9283
    ],

    "Context Precision": [

        0.75,

        0.82,

        0.90,

        0.94,

        0.9583
    ]
})

# =========================================================
# DATAFRAME
# =========================================================

st.dataframe(
    comparison_df,
    use_container_width=True
)

# =========================================================
# BAR CHART
# =========================================================

st.subheader("📊 Pipeline Improvement Comparison")

fig = px.bar(

    comparison_df,

    x="Pipeline",

    y=[
        "Faithfulness",
        "Answer Relevancy",
        "Context Precision"
    ],

    barmode="group",

    title="RAG Pipeline Evaluation Comparison"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# LINE CHART
# =========================================================

st.subheader("📈 Metric Improvement Trend")

fig = px.line(

    comparison_df,

    x="Pipeline",

    y=[
        "Faithfulness",
        "Answer Relevancy",
        "Context Precision"
    ],

    markers=True,

    title="Evaluation Metric Trends Across Pipeline Improvements"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# FINAL SUMMARY
# =========================================================

st.subheader("📝 Final Observations")

st.markdown(f"""
### Key Improvements

- **Faithfulness improved to {faithfulness:.4f}**
  after adding reranking and query rewriting.

- **Context Precision reached {context_precision:.4f}**
  using hybrid retrieval and reranking.

- **Answer Relevancy improved to {answer_relevancy:.4f}**
  through query optimization and better grounding.

### Final Architecture

```text
User Query
   ↓
Query Rewriting
   ↓
Hybrid Retrieval
   ↓
Reranking
   ↓
Groq LLM
   ↓
Grounded Response
```
""")