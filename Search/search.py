from langchain_ollama import ChatOllama
from Vector.vector_store import VectorStore
import os

class RagSearch:
    def __init__(self, persist_dir:str="faiss_store", embedding_model: str = "all-MiniLM-L6-v2",model_name="gemma3"):
        self.vector_store=VectorStore(persist_dir, embedding_model)
        faiss_file=os.path.join(persist_dir, "faiss.index")
        metadata_file=os.path.join(persist_dir, "pickel.index")
        if not (os.path.exists(faiss_file) and os.path.exists(metadata_file)):
            from document_upload.upload_doc import upload_document
            docs=upload_document("file")
            self.vector_store.build_from_document(docs)
        else:
            self.vector_store.load()
        self.llm=ChatOllama(model="gemma3", temperature=0.0)
        print(f"[INFO] Ollama LLM initialized: {model_name}")
        
    def search_and_summarize(self, query_llm:str, top_k:int=5):
        results=self.vector_store.embed_query(query_llm, top_k=top_k)
        texts=[r["metadata"].get("text", "") for r in results if r["metadata"]]
        context="\n\n".join(texts)
        if not context:
            return "No relevent document found"
        prompt=f"""summarize the following context for query:'{query_llm}' \n\nContext \n{context}\n\nSummary:"""
        response=self.llm.invoke([prompt])
        return response.content