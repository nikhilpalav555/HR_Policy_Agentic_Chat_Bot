import os
from pathlib import Path
from typing import List

from fastapi import UploadFile
from document_upload.upload_doc import upload_document
from Vector.vector_store import VectorStore
from langchain_ollama import ChatOllama
from ollama._types import ResponseError


class QASystem:
    def __init__(
        self,
        persist_dir: str,
        upload_dir: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gemma3",
    ):
        self.persist_dir = Path(persist_dir)
        self.upload_dir = Path(upload_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        self.vector_store = VectorStore(self.persist_dir.as_posix(), embedding_model)
        self.llm = ChatOllama(model=llm_model, temperature=0.2, num_predict=512)

        faiss_file = self.persist_dir / "faiss.index"
        metadata_file = self.persist_dir / "pickel.index"
        if faiss_file.exists() and metadata_file.exists():
            self.vector_store.load()
        else:
            self.vector_store.index = None
            self.vector_store.metadata = []

    async def save_uploaded_files(self, files: List[UploadFile]) -> List[Path]:
        saved = []
        for upload_file in files:
            destination = self.upload_dir / upload_file.filename
            content = await upload_file.read()
            destination.write_bytes(content)
            saved.append(destination)
        return saved

    def build_index_from_folder(self, folder: Path) -> int:
        documents = upload_document(str(folder))
        if not documents:
            return 0
        indexed_chunks = self.vector_store.build_from_document(documents)
        return indexed_chunks or len(documents)

    def get_answer(self, question: str, top_k: int = 5) -> str:
        if self.vector_store.index is None:
            return "No documents are indexed yet. Upload documents first."

        results = self.vector_store.embed_query(question, top_k=top_k)
        texts = [result["metadata"].get("text", "") for result in results if result.get("metadata")]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant document context found for the question."

        prompt = (
            "You are an expert assistant answering questions from provided document excerpts. "
            "Use the document text only, be detailed, and cite the information clearly.\n\n"
            f"Question: {question}\n\n"
            f"Document excerpts:\n{context}\n\n"
            "Give a thorough answer based on the documents. If the answer is not in the documents, say that the information is unavailable."
        )
        try:
            response = self.llm.invoke([prompt], reasoning=True)
        except ResponseError as exc:
            if "does not support thinking" in str(exc):
                response = self.llm.invoke([prompt])
            else:
                raise
        answer_text = getattr(response, "content", str(response))
        reasoning = response.additional_kwargs.get("reasoning_content") if hasattr(response, "additional_kwargs") else None
        if reasoning:
            answer_text += f"\n\n[Reasoning]\n{reasoning}"
        return answer_text
