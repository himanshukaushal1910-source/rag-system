# RAG System — Setup & Run Guide

Complete step-by-step guide to run the project using VS Code and the integrated PowerShell terminal.

---

## Prerequisites

Install these before starting. All are free.

| Tool | Download | Version |
|---|---|---|
| Python | https://www.python.org/downloads/ | 3.11 or 3.12 |
| Docker Desktop | https://www.docker.com/products/docker-desktop/ | Latest |
| VS Code | https://code.visualstudio.com/ | Latest |
| Git | https://git-scm.com/downloads | Latest |

In VS Code, install the **Python extension** (by Microsoft) from the Extensions panel.

---

## Step 1 — Open the project in VS Code

1. Open VS Code
2. Go to **File → Open Folder**
3. Select `D:\rag_system`
4. Open the integrated terminal: **Terminal → New Terminal** (or `Ctrl + \``)
5. Make sure the terminal type is **PowerShell** (shown in the top-right of the terminal panel). If it shows `cmd`, click the `+` dropdown and choose **PowerShell**.

---

## Step 2 — Create your `.env` file

In the PowerShell terminal:

```powershell
copy .env.example .env
```

Then open `.env` in VS Code (click the file in the Explorer panel) and fill in:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # your OpenAI key
API_KEY=my-secret-password                           # any password you choose for the UI login
```

Everything else in `.env` is already set to optimal defaults — leave it as-is.

---

## Step 3 — Start Qdrant (vector database) with persistent storage

This command starts Qdrant in Docker **with a volume mount** so your ingested data is never lost even if Docker restarts.

In the PowerShell terminal:

```powershell
docker run -d `
  --name qdrant `
  -p 6333:6333 `
  -p 6334:6334 `
  -v D:/rag_system/data/qdrant_storage:/qdrant/storage `
  qdrant/qdrant
```

Verify it is running:

```powershell
docker ps
```

You should see a container named `qdrant` with status `Up`.

> **Next time you restart your PC**, Qdrant won't start automatically. Run this to bring it back (data is preserved):
> ```powershell
> docker start qdrant
> ```

---

## Step 4 — Create and activate a Python virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Your terminal prompt will now show `(.venv)` at the start.

> If you get a script execution error, run this once and retry:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

## Step 5 — Install all dependencies

```powershell
pip install -e .
```

This installs everything from `pyproject.toml` including:
- FastAPI, Uvicorn
- OpenAI SDK
- Qdrant client
- Sentence Transformers (local reranker + chunking model)
- LangChain, LangGraph
- PyMuPDF, pdfplumber (PDF parsing)
- All other packages

> First run downloads ~2 GB of model weights. This only happens once.

---

## Step 6 — Ingest your PDFs

Your PDFs are already organized:

| Folder | PDFs | Action |
|---|---|---|
| `data/pdfs/papers` | 200 | Ingest now |
| `data/pdfs/papers_remaining` | 806 | Ingest later |

Run ingestion for the first 200 PDFs:

```powershell
python ingest.py
```

You will see a live progress bar showing:
- Chunks ingested
- PDFs/min rate
- Skipped (already done) / Failed counts
- ETA

> **Tip — faster first ingest:** If you want to skip figure description (GPT-4o vision calls) for speed, add `FIGURE_DESCRIPTION_ENABLED=false` to your `.env` before running. Re-enable later for visual query support.

> **Resume safety:** If ingestion is interrupted, just re-run `python ingest.py`. Already-ingested PDFs are detected by fingerprint and skipped automatically.

> **Batch size:** Default is 20 PDFs in parallel. Increase for faster ingestion:
> ```powershell
> python ingest.py 40
> ```

---

## Step 7 — Start the server

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

You should see output like:
```
INFO     Starting up RAG system
INFO     Qdrant ready
INFO     All components initialised — server ready
INFO     Uvicorn running on http://0.0.0.0:8000
```

> The `--reload` flag auto-restarts the server when you edit code. Remove it in production.

---

## Step 8 — Open the UI

Open your browser and go to:

```
http://localhost:8000
```

A **login dialog** will appear. Enter the `API_KEY` value you set in your `.env` file.

After login you will see the chat interface.

---

## Step 9 — Ask questions

Type any question in the chat box and press Enter or click Send.

The system will:
1. Classify your query type (factual / analytical / visual / table / code)
2. Generate HyDE + Step-Back + 3 RAG Fusion variants in parallel
3. Search Qdrant with all query variants
4. Rerank with the cross-encoder on your GPU
5. Expand context with sentence windows
6. Compress chunks to relevant sentences only
7. Generate a grounded answer with GPT-4o
8. Return the answer with numbered citations, faithfulness score, and rendered math/code/tables

---

## Step 10 — Ingest the remaining 806 PDFs (when ready)

Open `ingest.py` in VS Code and change line 49:

```python
# Change from:
PDF_DIR = Path(r"D:\rag_system\data\pdfs\papers")

# Change to:
PDF_DIR = Path(r"D:\rag_system\data\pdfs\papers_remaining")
```

Then run again:

```powershell
python ingest.py 40
```

Already-ingested PDFs from the first batch are automatically skipped.

---

## Daily Workflow

Every time you want to use the system after a PC restart:

```powershell
# 1. Start Qdrant (your data is preserved)
docker start qdrant

# 2. Open project in VS Code, open terminal, activate venv
.venv\Scripts\Activate.ps1

# 3. Start the server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Open browser at http://localhost:8000
```

---

## API Reference (optional — for direct access without the UI)

All endpoints require the `X-API-Key` header or a login cookie.

```powershell
# Login (sets cookie)
curl -X POST http://localhost:8000/api/login `
  -H "Content-Type: application/json" `
  -d '{"api_key": "your-api-key"}'

# Query
curl -X POST http://localhost:8000/api/query `
  -H "X-API-Key: your-api-key" `
  -H "Content-Type: application/json" `
  -d '{"query": "What is the main contribution of the paper?"}'

# Ingest a folder via API
curl -X POST http://localhost:8000/api/ingest `
  -H "X-API-Key: your-api-key" `
  -H "Content-Type: application/json" `
  -d '{"directory": "D:/rag_system/data/pdfs/papers"}'

# Check ingest job status
curl http://localhost:8000/api/ingest/JOB_ID_HERE `
  -H "X-API-Key: your-api-key"

# Health check
curl http://localhost:8000/health
```

---

## Troubleshooting

**`ModuleNotFoundError`**
You are not in the virtual environment. Run `.venv\Scripts\Activate.ps1` first.

**`Cannot connect to Qdrant`**
Qdrant is not running. Run `docker start qdrant`.

**`docker: command not found`**
Docker Desktop is not installed or not running. Start Docker Desktop from the Start Menu.

**`Set-ExecutionPolicy` error when activating venv**
Run this once in PowerShell as Administrator:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
```

**`RateLimitError` during ingestion**
Your OpenAI API tier is hitting rate limits. Reduce batch size:
```powershell
python ingest.py 10
```

**Server starts but UI shows blank page**
Clear browser cache or open in a private/incognito window.

**Low faithfulness scores on answers**
The answer is not well-grounded. Try:
- Being more specific in your question
- Adding the paper name: "According to [paper name], ..."
- Checking that the relevant PDF was ingested (look for it in citations)

---

## Project Structure (quick reference)

```
rag_system/
├── api/                  FastAPI server, routes, middleware, UI template
│   ├── main.py           App entry point and lifespan
│   ├── routes/           query.py, ingest.py, auth.py, metrics.py
│   └── templates/        index.html (the chat UI)
├── agent/                LangGraph agentic pipeline
│   ├── graph.py          Node wiring
│   └── nodes/            decomposer, retriever, generator, verifier
├── ingestion/            PDF parsing, chunking, embedding, upsert
├── retrieval/            Hybrid retriever, reranker, MMR, HyDE,
│                         RAG Fusion, Step-Back, Sentence Window,
│                         Contextual Compressor
├── data/
│   ├── pdfs/papers/              200 PDFs — ingest first
│   ├── pdfs/papers_remaining/    806 PDFs — ingest later
│   └── qdrant_storage/           Qdrant vector data (created by Docker)
├── config.py             All settings (reads from .env)
├── ingest.py             Ingestion CLI script
├── .env                  Your secrets (never commit this)
└── .env.example          Template — copy to .env
```
