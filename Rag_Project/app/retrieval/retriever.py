import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_chroma import Chroma
from app.ingestion.embedder import get_embedding_model

CHROMA_PATH = "chroma_db"


def load_vector_store():
    """
    Load persisted ChromaDB vector store.
    """

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )

    return vector_store


def retrieve_documents(query, k=5):
    """
    Retrieve top-k relevant chunks.
    """

    vector_store = load_vector_store()

    results = vector_store.similarity_search(
        query=query,
        k=k
    )

    return results


if __name__ == "__main__":

    query = "Explain SPI communication"

    results = retrieve_documents(query)

    print(f"\nTop Retrieved Chunks for Query: {query}\n")

    for i, doc in enumerate(results):

        print(f"\n===== RESULT {i+1} =====\n")

        print(doc.page_content[:1000])

        print("\nMETADATA:")

        print(doc.metadata)