import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from sentence_transformers import CrossEncoder

from app.retrieval.retriever import retrieve_documents


# Load reranker model
reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)


def rerank_documents(query, retrieved_docs, top_k=3):
    """
    Rerank retrieved documents using CrossEncoder.
    """

    pairs = [
        (query, doc.page_content)
        for doc in retrieved_docs
    ]

    scores = reranker.predict(pairs)

    scored_docs = list(zip(retrieved_docs, scores))

    ranked_docs = sorted(
        scored_docs,
        key=lambda x: x[1],
        reverse=True
    )

    top_docs = ranked_docs[:top_k]

    return top_docs


if __name__ == "__main__":

    query = "Explain SPI communication"

    retrieved_docs = retrieve_documents(query, k=5)

    reranked_docs = rerank_documents(
        query,
        retrieved_docs,
        top_k=3
    )

    print("\n===== RERANKED RESULTS =====\n")

    for i, (doc, score) in enumerate(reranked_docs):

        print(f"\n===== RESULT {i+1} =====")

        print(f"\nRERANK SCORE: {score:.4f}\n")

        print(doc.page_content[:1000])

        print("\nMETADATA:\n")

        print(doc.metadata)