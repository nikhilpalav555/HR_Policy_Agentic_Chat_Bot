import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure the local src package modules are importable when running from the repository root
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from qa_service import QASystem

app = FastAPI(title="RAG Document Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


def get_qa_service():
    if not hasattr(app.state, "qa_service"):
        app.state.qa_service = QASystem(
            persist_dir=str(BASE_DIR.parent / "faiss_store"),
            upload_dir=str(BASE_DIR / "files" / "uploaded"),
        )
    return app.state.qa_service


@app.get("/", response_class=HTMLResponse)
async def read_home():
    homepage = static_dir / "index.html"
    if not homepage.exists():
        raise HTTPException(status_code=404, detail="Frontend file not found")
    return HTMLResponse(homepage.read_text(encoding="utf-8"))


@app.post("/api/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    saved_files = await qa_service.save_uploaded_files(files)
    if not saved_files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    qa_service = get_qa_service()
    indexed_chunks = qa_service.build_index_from_folder(Path(qa_service.upload_dir))
    return {
        "status": "success",
        "uploaded_files": [path.name for path in saved_files],
        "indexed_chunks": indexed_chunks,
    }


class QuestionRequest(BaseModel):
    question: str


@app.post("/api/ask")
async def ask_question(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    qa_service = get_qa_service()
    answer = qa_service.get_answer(question)
    return {"question": question, "answer": answer}
    
    
    