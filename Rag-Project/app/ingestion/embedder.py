from langchain_huggingface import HuggingFaceEmbeddings

from chunker import split_documents
from loader import load_documents


def get_embedding_model():
    """
    Load embedding model.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en"
    )

    return embedding_model


if __name__ == "__main__":

    docs = load_documents()

    chunks = split_documents(docs)

    embedding_model = get_embedding_model()

    sample_embedding = embedding_model.embed_query(
        chunks[0].page_content
    )

    print(f"\nEmbedding Dimension: {len(sample_embedding)}\n")

    print(sample_embedding[:10])