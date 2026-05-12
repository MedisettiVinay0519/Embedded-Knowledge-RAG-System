import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


def get_llm():
    """
    Load Groq LLM for query rewriting.
    """

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

    return llm


def rewrite_query(query):
    """
    Rewrite user query for better retrieval.
    """

    prompt = ChatPromptTemplate.from_template(
    """
    You are a query rewriting assistant for an Embedded Systems and Operating Systems RAG system.

    The knowledge base contains:
    - UART communication
    - SPI communication
    - I2C communication
    - serial communication protocols
    - synchronous and asynchronous communication
    - baud rate and UART packet structure
    - MSP430 SPI interfacing
    - RTOS concepts
    - RTOS scheduling algorithms
    - task management in RTOS
    - semaphores and synchronization
    - context switching
    - interrupt handling
    - GPOS vs RTOS
    - kernel architecture
    - monolithic kernel
    - microkernel
    - hybrid kernel
    - operating system fundamentals
    - multiprocessing and multitasking
    - multithreading
    - process and thread scheduling
    - STM32 microcontrollers
    - ARM Cortex-M33
    - TrustZone security
    - SAU and IDAU
    - embedded debugging
    - embedded memory protection
    - secure and non-secure execution
    - embedded systems programming

    Rewrite the user query to improve:
    - semantic retrieval
    - BM25 retrieval
    - acronym expansion
    - technical clarity

    Rules:
    - Return ONLY the rewritten query
    - No explanations
    - No bullet points
    - Keep it concise
    - Prefer short keyword-rich rewritten queries
    - Preserve the embedded systems and operating systems context
    - Expand technical acronyms correctly
    - If the query is unrelated to embedded systems or operating systems, return the original query unchanged
    User Query:
    {query}
    """
)

    chain = prompt | get_llm()

    response = chain.invoke({
        "query": query
    })

    return response.content.strip()


if __name__ == "__main__":

    query = "What is SPI?"

    rewritten_query = rewrite_query(query)

    print("\n===== ORIGINAL QUERY =====\n")

    print(query)

    print("\n===== REWRITTEN QUERY =====\n")

    print(rewritten_query)