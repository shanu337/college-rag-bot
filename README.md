# 🎓 Hermes AI Academic Advisor
An automated, production-ready Local RAG (Retrieval-Augmented Generation) pipeline built to ingest college syllabi and textbook PDFs, allowing students to query course loads locally with zero data leaving their machine.

## 🛠️ Tech Stack
* **LLM Engine:** `qwen-hermes:latest` (14B Parameter Model via Ollama)
* **Embedding Engine:** `nomic-embed-text`
* **Orchestration:** LangChain (`langchain-ollama`)
* **Vector Database:** ChromaDB (Persistent Storage)
* **Frontend UI:** Streamlit
* **DevOps:** Docker & Docker Compose

## 🚀 Quick Start
1. Ensure Ollama is running locally with your models pulled.
2. Clone this repository and add your PDFs to the `docs/` folder.
3. Build and launch the containerized app:
   ```bash
   docker compose up --build
