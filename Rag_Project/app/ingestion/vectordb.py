from langchain_chroma import Chroma

from embedder import get_embedding_model
from chunker import split_documents
from loader import load_documents


CHROMA_PATH = "chroma_db"


def create_vector_store(chunks):
    """
    Create and persist Chroma vector database.
    """

    embedding_model = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )

    print("\nChromaDB vector store created successfully.\n")

    return vector_store


if __name__ == "__main__":

    docs = load_documents()

    chunks = split_documents(docs)

    vector_store = create_vector_store(chunks)

    print(f"Stored {len(chunks)} chunks in ChromaDB.")