import sys
from pathlib import Path

# =========================================================
# ROOT PATH
# =========================================================

sys.path.append(str(Path(__file__).resolve().parents[2]))

# =========================================================
# IMPORTS
# =========================================================

from langchain_community.document_loaders import PyPDFDirectoryLoader

# =========================================================
# LOAD DOCUMENTS
# =========================================================

def load_documents(data_path="Rag_Project/data"):

    data_dir = Path(data_path)

    if not data_dir.exists():

        raise FileNotFoundError(
            f"Data directory not found: {data_path}"
        )

    loader = PyPDFDirectoryLoader(str(data_dir))

    documents = loader.load()

    print(f"\nLoaded {len(documents)} pages from PDF documents.\n")

    return documents

# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    docs = load_documents()

    print("\n===== SAMPLE DOCUMENT =====\n")

    print(docs[0].page_content[:1000])

    print("\n===== METADATA =====\n")

    print(docs[0].metadata)