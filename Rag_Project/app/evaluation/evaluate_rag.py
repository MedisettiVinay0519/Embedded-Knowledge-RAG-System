import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app.generation.rag_chain import generate_response


load_dotenv()


# Groq evaluator LLM
evaluator_llm = LangchainLLMWrapper(
    ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )
)


# Local embedding model for RAGAS
embedding_model = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en"
    )
)


# Evaluation dataset
evaluation_data = [

    {
        "question": "Explain SPI communication",
        "ground_truth": "SPI is a synchronous serial communication protocol that uses clock signals for data transfer."
    },

    {
        "question": "What is UART?",
        "ground_truth": "UART is an asynchronous serial communication protocol used for serial data transmission."
    },

    {
        "question": "What is I2C communication?",
        "ground_truth": "I2C is a synchronous communication protocol that uses SDA and SCL lines for communication."
    },

    {
        "question": "What is an RTOS?",
        "ground_truth": "RTOS is a Real-Time Operating System designed for deterministic task scheduling."
    },

    {
        "question": "Explain hard real-time systems",
        "ground_truth": "Hard real-time systems require strict timing deadlines where missing a deadline may cause system failure."
    },

    {
        "question": "What is task scheduling in RTOS?",
        "ground_truth": "Task scheduling in RTOS determines which task executes based on priority and timing constraints."
    },

    {
        "question": "What is context switching?",
        "ground_truth": "Context switching is the process of saving and restoring CPU state between tasks or processes."
    },

    {
        "question": "What is deadlock in operating systems?",
        "ground_truth": "Deadlock is a condition where processes wait indefinitely for resources held by each other."
    },

    {
        "question": "Explain process scheduling",
        "ground_truth": "Process scheduling determines how CPU time is allocated among processes."
    },

    {
        "question": "What is TrustZone in ARM Cortex-M33?",
        "ground_truth": "TrustZone is a security feature in ARM Cortex-M processors that separates secure and non-secure execution environments."
    },

    {
        "question": "What is synchronous communication?",
        "ground_truth": "Synchronous communication uses a shared clock signal between devices during data transfer."
    },

    {
        "question": "What is asynchronous communication?",
        "ground_truth": "Asynchronous communication does not require a shared clock signal between communicating devices."
    },

    {
        "question": "Explain master-slave communication in SPI",
        "ground_truth": "SPI communication uses a master device to control communication with one or more slave devices."
    },

    {
        "question": "What is baud rate in UART?",
        "ground_truth": "Baud rate represents the speed of data transmission in UART communication."
    },

    {
        "question": "Explain semaphore in RTOS",
        "ground_truth": "A semaphore is a synchronization mechanism used for task coordination and resource sharing in RTOS."
    }
]


def prepare_evaluation_data():

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in evaluation_data:

        question = item["question"]

        result = generate_response(question)

        questions.append(question)

        answers.append(result["answer"])

        contexts.append(result["contexts"])

        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    return dataset


if __name__ == "__main__":

    eval_dataset = prepare_evaluation_data()

    print("\nRunning RAG Evaluation...\n")

    results = evaluate(
        dataset=eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
        ],
        llm=evaluator_llm,
        embeddings=embedding_model
    )

    print("\n===== EVALUATION RESULTS =====\n")

    print(results)