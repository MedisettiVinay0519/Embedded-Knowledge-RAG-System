import sys
from pathlib import Path

# =========================================================
# ROOT PATH
# =========================================================

sys.path.append(str(Path(__file__).resolve().parents[2]))

# =========================================================
# IMPORTS
# =========================================================

from langchain_chroma import Chroma

from app.ingestion.embedder import get_embedding_model

# =========================================================
# CHROMA CONFIG
# =========================================================

CHROMA_PATH = "chroma_db"

# =========================================================
# CREATE VECTOR STORE
# =========================================================

def create_vector_store(chunks):

    embedding_model = get_embedding_model()

    vector_store = Chroma.from_documents(

        documents=chunks,

        embedding=embedding_model,

        persist_directory=CHROMA_PATH
    )

    print("\n✅ Vector database created successfully.")

    print(f"\nStored {len(chunks)} chunks in ChromaDB.")

    return vector_store

# =========================================================
# LOAD VECTOR STORE
# =========================================================

def load_vector_store():

    embedding_model = get_embedding_model()

    vector_store = Chroma(

        persist_directory=CHROMA_PATH,

        embedding_function=embedding_model
    )

    return vector_store