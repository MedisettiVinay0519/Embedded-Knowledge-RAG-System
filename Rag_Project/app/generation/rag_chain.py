import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# =========================================================
# ROOT PATH
# =========================================================

sys.path.append(str(Path(__file__).resolve().parents[2]))

# =========================================================
# IMPORTS
# =========================================================

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from app.query_rewrite.query_rewriter import rewrite_query
from app.retrieval.hybrid_retriever import hybrid_retrieve
from app.reranking.reranking import rerank_documents

# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

# =========================================================
# LLM
# =========================================================

def get_llm():

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

    return llm

# =========================================================
# MAIN RAG PIPELINE
# =========================================================

def generate_response(query):

    start_total = time.time()

    # =====================================================
    # QUERY REWRITE
    # =====================================================

    start = time.time()

    rewritten_query = rewrite_query(query)

    rewrite_time = time.time() - start

    # =====================================================
    # DOMAIN GUARDRAIL
    # =====================================================

    embedded_keywords = [

        "spi",
        "uart",
        "i2c",
        "rtos",
        "arm",
        "stm32",
        "trustzone",
        "kernel",
        "interrupt",
        "microcontroller",
        "embedded",
        "scheduling",
        "semaphore",
        "context switching",
        "communication protocol",
        "operating system"
    ]

    query_lower = rewritten_query.lower()

    if not any(keyword in query_lower for keyword in embedded_keywords):

        return {

            "original_query": query,

            "rewritten_query": rewritten_query,

            "answer": (
                "This RAG system is specialized for Embedded Systems and "
                "Operating Systems topics. Please ask questions related to "
                "embedded systems, RTOS, ARM Cortex, communication protocols, "
                "or operating systems."
            ),

            "contexts": [],

            "source_data": [],

            "latency": {

                "rewrite_time": round(rewrite_time, 2),

                "retrieval_time": 0,

                "rerank_time": 0,

                "generation_time": 0,

                "total_time": round(time.time() - start_total, 2)
            }
        }

    # =====================================================
    # HYBRID RETRIEVAL
    # =====================================================

    start = time.time()

    retrieved_docs = hybrid_retrieve(
        query=rewritten_query,
        k=8
    )

    retrieval_time = time.time() - start

    # =====================================================
    # RERANKING
    # =====================================================

    start = time.time()

    reranked_docs = rerank_documents(
        query=rewritten_query,
        retrieved_docs=retrieved_docs,
        top_k=4
    )

    rerank_time = time.time() - start

    # =====================================================
    # FINAL DOCUMENTS
    # =====================================================

    final_docs = []

    source_data = []

    for doc, score in reranked_docs:

        final_docs.append(doc)

        source = doc.metadata.get("source", "Unknown")

        page = doc.metadata.get("page_label", "Unknown")

        source_data.append({

            "source": Path(source).name,

            "page": page,

            "score": round(score, 4),

            "content": doc.page_content
        })

    # =====================================================
    # CONTEXT
    # =====================================================

    context = "\n\n".join(
        [doc.page_content for doc in final_docs]
    )

    # =====================================================
    # PROMPT
    # =====================================================

    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert Embedded Systems and Operating Systems assistant.

        Answer the user's question using ONLY the provided context.

        STRICT RULES:
        - Do NOT add external knowledge
        - Do NOT hallucinate
        - Keep answers concise and factual
        - If information is unavailable say:
          "I could not find relevant information in the provided documents."

        Context:
        {context}

        User Question:
        {question}
        """
    )

    chain = prompt | get_llm()

    # =====================================================
    # GENERATION
    # =====================================================

    start = time.time()

    response = chain.invoke({
        "context": context,
        "question": query
    })

    generation_time = time.time() - start

    total_time = time.time() - start_total

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    return {

        "original_query": query,

        "rewritten_query": rewritten_query,

        "answer": response.content,

        "contexts": [doc.page_content for doc in final_docs],

        "source_data": source_data,

        "latency": {

            "rewrite_time": round(rewrite_time, 2),

            "retrieval_time": round(retrieval_time, 2),

            "rerank_time": round(rerank_time, 2),

            "generation_time": round(generation_time, 2),

            "total_time": round(total_time, 2)
        }
    }

# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    query = "Explain SPI communication"

    result = generate_response(query)

    print("\n==============================")
    print(" FINAL GENERATED ANSWER ")
    print("==============================\n")

    print(result["answer"])

    print("\n==============================")
    print(" SOURCES ")
    print("==============================\n")

    for item in result["source_data"]:

        print(
            f"{item['source']} | "
            f"Page {item['page']} | "
            f"Score: {item['score']}"
        )