# LawSearch AI

**LawSearch AI** is a Retrieval-Augmented Generation (RAG) system for querying U.S. federal appropriations bills using natural language. Built with LangChain and OpenAI, it enables detailed analysis of legislative text through semantic search and contextual summarization.

## Overview

This project addresses the challenge of navigating complex federal budget documents by creating a searchable knowledge base that responds to natural language queries. The system processes raw legislative text into searchable chunks, indexes them in vector databases, and enables concurrent retrieval across multiple bill divisions.

## Features

- **Document Processing**: Parses federal appropriations bills into logical divisions
- **Vector Indexing**: Creates persistent searchable embeddings using Chroma
- **Concurrent Search**: Uses asyncio for parallel querying across document sections
- **Context-Aware Responses**: Synthesizes information using GPT-4o models
- **Persistent Storage**: Maintains embeddings for future queries without reprocessing

## Project Structure

```bash
lawSearch/
├── data/                   # Raw HTML appropriations bills
├── db/                     # Chroma vectorstore directories
├── src/
│   ├── config.py           # Centralized constants (paths, model settings)
│   ├── ingest.py           # Ingests bills into separate vectorstores
│   └── query.py            # Queries vectorstores concurrently
├── .env                    # Store your OpenAI API Key here
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- OpenAI API key
- 4GB+ available storage for vector databases

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/lawsearch.git
   cd lawsearch
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your API key:
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

### Usage

1. **Prepare your data**:
   Place appropriations bill HTML files in `data/bills/`

2. **Process documents**:
   ```bash
   python -m src.ingest
   ```

3. **Run queries**:
   ```bash
   python -m src.query "How much funding is allocated to renewable energy research?"
   ```

## Example Queries

- "What funding is allocated to veterans' healthcare services?"
- "Compare defense and education spending in this appropriation."
- "Find all grants related to climate change research."
- "Summarize funding for border security initiatives."

## Technical Implementation

LawSearch AI uses a multi-stage pipeline:

1. **Document Processing**: HTML parsing with BeautifulSoup, division into logical sections
2. **Text Chunking**: Recursive character splitting to create semantically-meaningful segments
3. **Embedding Generation**: OpenAI's embedding models create vector representations
4. **Retrieval**: k-NN search against embeddings to find relevant text chunks
5. **Synthesis**: LLM-based summarization and refinement using retrieved contexts

## License

MIT License © 2024 Ryan Mahshie
