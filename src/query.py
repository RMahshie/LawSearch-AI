#!/usr/bin/env python3
import asyncio
import os
import time
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from src.config import VECTORSTORE_DIR, EMBEDDING_MODEL, LLM_INGEST, LLM_SUMMARY

# 1️⃣ Instantiate LLMs
qa_llm      = ChatOpenAI(model_name=LLM_INGEST, temperature=0)
summary_llm = ChatOpenAI(model_name=LLM_SUMMARY)

# 2️⃣ Load all per-division vectorstores
#    (each directory under VECTORSTORE_DIR is one division's DB)
embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)

def load_vectorstores():
    stores = {}
    for name in os.listdir(VECTORSTORE_DIR):
        path = os.path.join(VECTORSTORE_DIR, name)
        if os.path.isdir(path):
            stores[name] = Chroma(
                persist_directory=path,
                embedding_function=embedder,
            )
    return stores

division_stores = load_vectorstores()

# 3️⃣ Prepare PromptTemplates
refine_prompt = PromptTemplate(
    template="""
You have an existing answer and additional context. Refine the existing answer to incorporate the new context.

---------------
Existing answer:
{existing_answer}
---------------
Additional context:
{context_str}
---------------
Question:
{question}
""",
    input_variables=["existing_answer", "context_str", "question"],
)
summarize_prompt = PromptTemplate(
    template="""
Summarize each division’s findings in bullets with amounts, then give an overall summary.

Findings by Division:
{formatted_findings}

Overall Summary:
""",
    input_variables=["formatted_findings"],
)

# 4️⃣ Concurrency control
semaphore = asyncio.Semaphore(15)

async def query_division(label: str, store: Chroma, question: str):
    async with semaphore:
        retriever = store.as_retriever(search_kwargs={"k": 10})
        chain = RetrievalQA.from_chain_type(
            llm=qa_llm,
            retriever=retriever,
            chain_type="refine",
            chain_type_kwargs={"refine_prompt": refine_prompt},
        )
        answer = await asyncio.to_thread(chain.run, question)
        return label, answer

async def query_all(question: str):
    tasks = [query_division(lbl, st, question) for lbl, st in division_stores.items()]
    results = await asyncio.gather(*tasks)
    return dict(results)

def summarize_results(results: dict[str, str]) -> str:
    formatted = "\n".join(f"🔹 {lbl}:\n{ans}" for lbl, ans in results.items())
    message = summarize_prompt.format(formatted_findings=formatted)
    return summary_llm.invoke([HumanMessage(content=message)]).content

async def main():
    question = input("Your question: ")

    start_time = time.time()

    raw_results = await query_all(question)
    summary = summarize_results(raw_results)


    end_time = time.time()
    time_elapsed = end_time - start_time

    print("\n=== Summary ===\n", summary)
    print("\n=== Details ===")
    for lbl, ans in raw_results.items():
        print(f"\n--- {lbl} ---\n{ans}")

    print(f"Time taken: {time_elapsed:.2f} seconds")
if __name__ == '__main__':
    asyncio.run(main())
