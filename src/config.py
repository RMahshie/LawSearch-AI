import os

# Vectorstore persistence
BASE_DIR       = os.path.dirname(os.path.dirname(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data", "bills")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "db", "chroma")

# API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embedding & LLM config
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_INGEST      = "gpt-4o-mini"
LLM_SUMMARY     = "o4-mini"

# Chunking parameters
CHUNK_SIZE      = 20000
CHUNK_OVERLAP   = 500
