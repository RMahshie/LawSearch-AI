#!/usr/bin/env python3
import re
import os
import shutil
import glob
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.config import DATA_DIR, VECTORSTORE_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Regex pattern for division headers
division_pattern = re.compile(
    r'^\s*DIVISION\s+([A-Z])\s*--\s*(?:(OTHER MATTERS)|((?:(?!APPROPRIATIONS).)+))',
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)

def process_file(path: str) -> dict[str, str]:
    """Split a single HTML bill into division-labeled text chunks."""
    html = open(path, encoding="utf-8").read()
    soup = BeautifulSoup(html, "html.parser")
    full_text = soup.get_text(separator="\n")
    clean_text = re.sub(r'<<NOTE:[^>]+>>', '', full_text)

    matches = list(division_pattern.finditer(clean_text))
    if not matches:
        return {}

    # Map division letter to agency name
    division_names: dict[str, str] = {}
    for m in matches:
        div = m.group(1).upper()
        raw = m.group(2) or m.group(3) or ''
        division_names[div] = ' '.join(raw.split())

    # Build list of (start, letter) and slice
    headers = sorted((m.start(), m.group(1).upper()) for m in matches)
    chunks: dict[str, str] = {}
    for i, (start, div) in enumerate(headers):
        end = headers[i+1][0] if i+1 < len(headers) else len(clean_text)
        chunk_text = clean_text[start:end].strip()
        label = f"{os.path.basename(path)} - Division {div} - {division_names[div]}"
        chunks[label] = chunk_text
    return chunks


def ingest_all():
    """Ingest all HTML files in DATA_DIR into per-division Chroma databases."""
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    html_files = glob.glob(os.path.join(DATA_DIR, '*.html'))
    if not html_files:
        raise RuntimeError(f"No HTML files found in {DATA_DIR}")

    all_chunks: dict[str, str] = {}
    for file in html_files:
        divisions = process_file(file)
        print(f"Found {len(divisions)} divisions in {os.path.basename(file)}")
        all_chunks.update(divisions)

    # Initialize embedder and text splitter
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # Create or refresh each division's vectorstore
    for label, text in all_chunks.items():
        docs = splitter.split_documents([
            Document(page_content=text, metadata={'division': label})
        ])
        safe_name = re.sub(r'[^\w]+', '_', label)
        db_path = os.path.join(VECTORSTORE_DIR, safe_name)
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        os.makedirs(db_path, exist_ok=True)

        Chroma.from_documents(
            docs,
            embedder,
            persist_directory=db_path
        )
        print(f"Created Chroma DB for '{label}' at '{db_path}'")


if __name__ == '__main__':
    ingest_all()
