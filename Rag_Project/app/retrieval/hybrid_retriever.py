import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from app.ingestion.loader import load_documents
from app.ingestion.chunker import split_documents
from app.ingestion.embedder import get_embedding_model


CHROMA_PATH = "chroma_db"


def load_vector_store():
    """
    Load ChromaDB vector store.
    """

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )

    return vector_store


def get_bm25_retriever():
    """
    Create BM25 retriever from chunks.
    """

    docs = load_documents()

    chunks = split_documents(docs)

    bm25_retriever = BM25Retriever.from_documents(chunks)

    bm25_retriever.k = 5

    return bm25_retriever


def hybrid_retrieve(query, k=5):
    """
    Hybrid Retrieval:
    Vector Search + BM25 Search
    """

    # Vector Retrieval
    vector_store = load_vector_store()

    vector_results = vector_store.similarity_search(
        query=query,
        k=k
    )

    # BM25 Retrieval
    bm25_retriever = get_bm25_retriever()

    bm25_results = bm25_retriever.invoke(query)

    # Merge Results
    combined_results = vector_results + bm25_results

    # Remove duplicates
    unique_docs = []

    seen_content = set()

    for doc in combined_results:

        content = doc.page_content

        if content not in seen_content:

            unique_docs.append(doc)

            seen_content.add(content)

    return unique_docs[:k]


if __name__ == "__main__":

    query = "Explain SPI communication"

    results = hybrid_retrieve(query)

    print(f"\n===== HYBRID RETRIEVAL RESULTS =====\n")

    for i, doc in enumerate(results):

        print(f"\n===== RESULT {i+1} =====\n")

        print(doc.page_content[:1000])

        print("\nMETADATA:\n")

        print(doc.metadata)