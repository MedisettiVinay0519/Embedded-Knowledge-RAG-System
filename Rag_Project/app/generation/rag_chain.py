import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from app.retrieval.retriever import retrieve_documents


load_dotenv()


def get_llm():

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

    return llm


def generate_response(query):

    retrieved_docs = retrieve_documents(query)

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using ONLY the provided context.

        Context:
        {context}

        Question:
        {question}
        """
    )

    chain = prompt | get_llm()

    response = chain.invoke({
        "context": context,
        "question": query
    })

    return response.content


if __name__ == "__main__":

    query = "Explain SPI communication"

    answer = generate_response(query)

    print("\n===== GENERATED ANSWER =====\n")

    print(answer)