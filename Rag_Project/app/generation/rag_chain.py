import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from app.query_rewrite.query_rewriter import rewrite_query
from app.retrieval.hybrid_retriever import hybrid_retrieve
from app.reranking.reranking import rerank_documents


load_dotenv()


def get_llm():
    """
    Load Groq LLM.
    """

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

    return llm


def generate_response(query):
    """
    Final Advanced RAG Pipeline:
    Query Rewrite
    → Hybrid Retrieval
    → Reranking
    → Groq LLM Generation
    """

    # =========================================================
    # STEP 1: QUERY REWRITING
    # =========================================================

    rewritten_query = rewrite_query(query)

    print("\n===== REWRITTEN QUERY =====\n")

    print(rewritten_query)

    # =========================================================
    # STEP 2: HYBRID RETRIEVAL
    # =========================================================

    retrieved_docs = hybrid_retrieve(
        query=rewritten_query,
        k=8
    )

    print(f"\nRetrieved {len(retrieved_docs)} documents.")

    # =========================================================
    # STEP 3: RERANKING
    # =========================================================

    reranked_docs = rerank_documents(
        query=rewritten_query,
        retrieved_docs=retrieved_docs,
        top_k=4
    )

    print("\n===== TOP RERANKED DOCUMENTS =====\n")

    final_docs = []

    for i, (doc, score) in enumerate(reranked_docs):

        print(f"Document {i+1} | Score: {score:.4f}")

        final_docs.append(doc)

    # =========================================================
    # STEP 4: CONTEXT CREATION
    # =========================================================

    context = "\n\n".join(
        [doc.page_content for doc in final_docs]
    )

    # =========================================================
    # STEP 5: PROMPT TEMPLATE
    # =========================================================

    
    prompt = ChatPromptTemplate.from_template(
    """
    You are an expert Embedded Systems and Operating Systems assistant.

    Answer the user's question using ONLY the provided context.

    STRICT RULES:
    - Do NOT add external knowledge
    - Do NOT make assumptions
    - Do NOT hallucinate
    - If information is missing, say:
      "I could not find relevant information in the provided documents."
    - Keep answers concise and factual
    - Use only facts present in the context
    - Do not explain beyond the retrieved content

    Context:
    {context}

    User Question:
    {question}
    """
)
    

    # =========================================================
    # STEP 6: LLM CHAIN
    # =========================================================

    chain = prompt | get_llm()

    response = chain.invoke({
        "context": context,
        "question": query
    })

    # =========================================================
    # STEP 7: SOURCE CITATIONS
    # =========================================================

    sources = []

    for doc in final_docs:

        metadata = doc.metadata

        source = metadata.get("source", "Unknown Source")

        page = metadata.get("page_label", "Unknown Page")

        sources.append(f"{Path(source).name} | Page {page}")

    # Remove duplicates
    sources = list(set(sources))

    # =========================================================
    # FINAL RESPONSE
    # =========================================================

    return {
        "original_query": query,
        "rewritten_query": rewritten_query,
        "answer": response.content,
        "contexts": [doc.page_content for doc in final_docs],
        "sources": sources
    }


if __name__ == "__main__":

    query = "What is SPI?"

    result = generate_response(query)

    print("\n==============================")
    print(" FINAL GENERATED ANSWER ")
    print("==============================\n")

    print(result["answer"])

    print("\n==============================")
    print(" SOURCES ")
    print("==============================\n")

    for source in result["sources"]:

        print(f"- {source}")