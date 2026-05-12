from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.ingestion.loader import load_documents


def split_documents(documents):
    """
    Split documents into smaller chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    print(f"\nCreated {len(chunks)} chunks.\n")

    return chunks


if __name__ == "__main__":

    docs = load_documents()

    chunks = split_documents(docs)

    print("===== SAMPLE CHUNK =====\n")

    print(chunks[0].page_content)

    print("\n===== CHUNK METADATA =====\n")

    print(chunks[0].metadata)