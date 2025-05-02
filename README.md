
# LawSearch AI

**LawSearch AI** is a Retrieval-Augmented Generation (RAG) system that allows users to query complex U.S. federal appropriations bills using natural language. It leverages LLMs and LangChain to chunk, embed, store, and asynchronously query legislative documents, enabling fast and interpretable budget analysis.

---

## 🧠 Features

- ⚖️ **Appropriations Bill Parser**: Extracts and segments federal legislation (e.g., Consolidated Appropriations Act) by division.
- 🧩 **Recursive Chunking**: Uses LangChain's `RecursiveCharacterTextSplitter` to prepare LLM-sized context windows.
- 📦 **Multi-VectorStore Indexing**: Creates a persistent `Chroma` vector database per bill division.
- 🔍 **Concurrent Asynchronous Querying**: Searches all divisions simultaneously using asyncio for maximum performance.
- 📝 **LLM-Powered Summarization**: Synthesizes results using `refine` chain logic and generates an executive summary.
- 🔐 **OpenAI API Integration**: Uses GPT-3.5 and gpt-4o-mini to balance performance and cost.

---

## 📁 Project Structure

```bash
lawSearch/
├── data/                   # Raw HTML appropriations bills
├── db/                     # Chroma vectorstore directories
├── src/
│   ├── config.py           # Centralized constants (paths, model settings)
│   ├── ingest.py           # Ingests bills into separate vectorstores
│   └── query.py            # Queries vectorstores concurrently
├── .env                    # (Optional) contains your OpenAI API key
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure you have Python 3.10+ and a working [OpenAI API key](https://platform.openai.com/account/api-keys).

### 2. Set Up API Key

Export it in your environment:

```bash
export OPENAI_API_KEY=your-key-here
```

Or use a `.env` file:

```env
OPENAI_API_KEY=your-key-here
```

### 3. Ingest Legislative Files

Place your raw `.html` bills in `data/bills/`, then run:

```bash
python -m src.ingest
```

This will parse, split, embed, and persist per-division databases into `db/chroma/`.

### 4. Run a Query

```bash
python -m src.query
```

You'll be prompted to ask a question like:

```
how much funding did FEMA receive?
```

The tool will return per-division answers + a synthesized summary using a second LLM.

---

## 📌 Example Questions

- `"How much money is allocated to FEMA?"`
- `"What funding is available for cybersecurity initiatives?"`
- `"Summarize all environmental spending in these bills."`

---

## ⚙️ Tech Stack

- [LangChain](https://www.langchain.com/)
- [OpenAI API](https://platform.openai.com/)
- [Chroma](https://www.trychroma.com/)
- [asyncio](https://docs.python.org/3/library/asyncio.html)
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)

---

## 📝 TODOs

- [ ] Add support for PDF bill parsing
- [ ] Add web UI for interaction
- [ ] Add tests for ingestion and parsing logic
- [ ] Switch to LangChain Expression Language (LCEL)

---

## 📄 License

MIT License © 2024 Ryan Mahshie
