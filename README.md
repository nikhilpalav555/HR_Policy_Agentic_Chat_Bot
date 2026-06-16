# RAG Document Q&A

A small retrieval-augmented generation (RAG) project that indexes uploaded documents and answers user questions using a local Ollama LLM (`gemma3`). The backend is built with FastAPI and the frontend is a simple HTML page served from the same app.

## Features

- Upload documents in PDF, TXT, and CSV formats
- Build a Faiss vector index from uploaded documents
- Query the document collection using an LLM
- Simple web UI for upload and Q&A
- FastAPI REST endpoints for upload and ask operations

## Project Structure

- `requirements.txt` — Python dependencies
- `faiss_store/` — persistent Faiss index and metadata
- `src/main.py` — FastAPI app and routes
- `src/qa_service.py` — QA service for indexing and querying
- `src/static/index.html` — frontend UI
- `src/document_upload/upload_doc.py` — document ingestion functions
- `src/Embeddings/embeddings.py` — text chunking and embedding logic
- `src/Vector/vector_store.py` — Faiss vector store management
- `src/Search/search.py` — RAG search example logic

## Setup

1. Create and activate your virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. If you are using Ollama and want faster model downloads, set `HF_TOKEN` in your environment.

## Run the app

From the project root:

```powershell
cd src
..\venv\Scripts\python.exe -m uvicorn main:app --reload
```

Open the browser at:

```text
http://127.0.0.1:8000/
```

## API Endpoints

### `POST /api/upload`
Upload documents and build the vector index.

- Request: `multipart/form-data`
- Field: `files` (one or more files)
- Supported file types: `.pdf`, `.txt`, `.csv`

Response example:

```json
{
  "status": "success",
  "uploaded_files": ["example.pdf"],
  "indexed_chunks": 12
}
```

### `POST /api/ask`
Ask a question about the uploaded documents.

- Request: JSON body with `question`

Example request:

```json
{
  "question": "What is the main objective of the uploaded document?"
}
```

Example response:

```json
{
  "question": "What is the main objective of the uploaded document?",
  "answer": "..."
}
```

## Frontend UI

The UI is served from `src/static/index.html` and provides:

- file upload field
- upload button
- question textbox
- answer display area

## Notes

- The first request may take longer if the Faiss index is built or loaded for the first time.
- The current LLM configuration uses `gemma3` via `langchain-ollama`.
- If the app fails due to model or reasoning mode restrictions, edit `src/qa_service.py` to adjust Ollama parameters.
- This version also includes tool-enabled reasoning: the LLM can call a document search tool, a calculator tool, and a current date tool.

## Tool-based reasoning support

The project now supports tools in the RAG pipeline. The current tool names are:

- `search_documents` — searches the indexed documents and returns relevant excerpts
- `calculator` — evaluates math expressions
- `check_current_date` — returns today’s date

If you add more tools to a normal RAG pipeline, use clear, descriptive names such as:

- `document_search` or `search_documents`
- `retrieve_documents` or `document_retrieval`
- `qa_search_tool` or `rag_search_tool`
- `extract_summary` or `summarize_documents`

Good naming makes it easier for the LLM to choose the correct tool and for other developers to understand the pipeline.

## Troubleshooting

- If file upload fails, ensure `python-multipart` is installed.
- If the server does not start, verify the current working directory is `src` when running Uvicorn.
- If the model returns short answers, increase `num_predict` or adjust the prompt in `src/qa_service.py`.

## License

This project is provided as-is for experimentation and local RAG use cases.
