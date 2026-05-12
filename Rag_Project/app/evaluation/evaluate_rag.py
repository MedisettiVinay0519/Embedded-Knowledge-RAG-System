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


# =========================================================
# GROQ EVALUATION LLM
# =========================================================

evaluator_llm = LangchainLLMWrapper(
    ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

embedding_model = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en"
    )
)


# =========================================================
# EVALUATION DATASET
# =========================================================

evaluation_data = [

    {
        "question": "Explain SPI communication protocol",
        "ground_truth": "SPI is a synchronous serial communication protocol that uses a shared clock signal for data transfer between master and slave devices."
    },

    {
        "question": "What is UART communication?",
        "ground_truth": "UART is an asynchronous serial communication protocol that transfers data without a clock signal."
    },

    {
        "question": "Explain I2C communication",
        "ground_truth": "I2C is a synchronous serial communication protocol that uses SDA and SCL lines for communication between devices."
    },

    {
        "question": "Differentiate synchronous and asynchronous communication",
        "ground_truth": "Synchronous communication uses a shared clock signal while asynchronous communication does not require a clock signal."
    },

    {
        "question": "What is baud rate in UART?",
        "ground_truth": "Baud rate represents the speed of data transmission in UART communication and is measured in bits per second."
    },

    {
        "question": "Explain UART packet structure",
        "ground_truth": "UART packet structure contains a start bit, data bits, optional parity bit, and stop bits."
    },

    {
        "question": "What are the advantages of UART?",
        "ground_truth": "UART requires only two wires, does not need a clock signal, and supports error checking using parity bits."
    },

    {
        "question": "What are the disadvantages of UART?",
        "ground_truth": "UART supports limited frame size, does not support multiple masters, and requires matching baud rates."
    },

    {
        "question": "Explain master-slave communication in SPI",
        "ground_truth": "SPI communication uses a master device that controls communication with one or more slave devices."
    },

    {
        "question": "What is an RTOS?",
        "ground_truth": "RTOS is a Real-Time Operating System designed to process data within strict timing constraints."
    },

    {
        "question": "Explain hard real-time systems",
        "ground_truth": "Hard real-time systems require tasks to complete within strict deadlines or system failure may occur."
    },

    {
        "question": "Differentiate GPOS and RTOS",
        "ground_truth": "GPOS focuses on general-purpose computing while RTOS focuses on deterministic and time-constrained task execution."
    },

    {
        "question": "Explain task scheduling in RTOS",
        "ground_truth": "Task scheduling in RTOS determines task execution order based on timing and priority constraints."
    },

    {
        "question": "What is context switching?",
        "ground_truth": "Context switching is the process of saving and restoring task or process state during CPU switching."
    },

    {
        "question": "Explain semaphores in RTOS",
        "ground_truth": "Semaphores are synchronization mechanisms used for task coordination and resource sharing in RTOS."
    },

    {
        "question": "What is preemptive scheduling?",
        "ground_truth": "Preemptive scheduling allows a higher priority task to interrupt and suspend a currently running task."
    },

    {
        "question": "What is round robin scheduling?",
        "ground_truth": "Round robin scheduling allocates CPU time slices equally among tasks in cyclic order."
    },

    {
        "question": "Explain monolithic kernel",
        "ground_truth": "A monolithic kernel contains all major operating system services within the kernel space."
    },

    {
        "question": "Explain microkernel architecture",
        "ground_truth": "A microkernel provides only essential services in kernel space while other services run in user space."
    },

    {
        "question": "What is a hybrid kernel?",
        "ground_truth": "A hybrid kernel combines features of monolithic kernels and microkernels."
    },

    {
        "question": "What is process scheduling in operating systems?",
        "ground_truth": "Process scheduling determines how CPU resources are allocated among multiple processes."
    },

    {
        "question": "Differentiate process and thread",
        "ground_truth": "A process is a program under execution while a thread is an independent execution path within a process."
    },

    {
        "question": "What is multithreading?",
        "ground_truth": "Multithreading allows multiple threads to execute concurrently within a single process."
    },

    {
        "question": "What is multitasking?",
        "ground_truth": "Multitasking enables multiple tasks or processes to execute seemingly simultaneously using context switching."
    },

    {
        "question": "Explain TrustZone in ARM Cortex-M33",
        "ground_truth": "TrustZone is a security feature in ARM Cortex-M33 that separates secure and non-secure execution environments."
    },

    {
        "question": "What are secure and non-secure states in TrustZone?",
        "ground_truth": "TrustZone divides execution into secure and non-secure states to provide hardware-level security isolation."
    },

    {
        "question": "What is SAU in TrustZone?",
        "ground_truth": "SAU or Security Attribution Unit defines secure and non-secure memory regions in TrustZone."
    },

    {
        "question": "What is IDAU in TrustZone?",
        "ground_truth": "IDAU or Implementation Defined Attribution Unit provides hardware-defined security memory partitioning."
    },

    {
        "question": "Explain invasive and non-invasive debugging",
        "ground_truth": "Invasive debugging allows processor control and modification while non-invasive debugging only observes processor behavior."
    },

    {
        "question": "What is secure debug access in ARM TrustZone?",
        "ground_truth": "Secure debug access provides visibility and debugging access to both secure and non-secure software regions."
    }
]
# =========================================================
# PREPARE EVALUATION DATA
# =========================================================

def prepare_evaluation_data():

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in evaluation_data:

        question = item["question"]

        print(f"\nProcessing Question: {question}")

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


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("\n===================================")
    print(" RUNNING ADVANCED RAG EVALUATION ")
    print("===================================\n")

    eval_dataset = prepare_evaluation_data()

    print("\n===================================")
    print(" EVALUATING WITH RAGAS ")
    print("===================================\n")

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

    print("\n===================================")
    print(" FINAL EVALUATION RESULTS ")
    print("===================================\n")

    print(results)