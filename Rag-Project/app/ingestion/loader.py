from langchain_community.document_loaders import PyPDFDirectoryLoader
from pathlib import Path


# Get project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Data directory path
DATA_PATH = BASE_DIR / "data"


def load_documents():
    """
    Load all PDF documents recursively from the data directory.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_PATH}")

    loader = PyPDFDirectoryLoader(
        path=str(DATA_PATH),
        glob="**/*.pdf"
    )

    documents = loader.load()

    print(f"\nLoaded {len(documents)} pages from PDF documents.\n")

    return documents


if __name__ == "__main__":

    docs = load_documents()

    print("===== SAMPLE DOCUMENT =====\n")

    print(docs[0].page_content[:1000])

    print("\n===== METADATA =====\n")

    print(docs[0].metadata)