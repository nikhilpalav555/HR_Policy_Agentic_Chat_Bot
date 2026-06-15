import faiss
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
import numpy as np
from typing import List, Any
import os
from Embeddings.embeddings import Embeddings
import pickle


class VectorStore:
    def __init__(self,persist_dir:str="faiss_store", embedding_model:str="all-MiniLM-L6-v2", chunk_size=1000, chunk_overlay=200 ):
        self.metadata=[]
        self.persist_dir=persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index=None
        self.chunk_size=chunk_size
        self.chunk_overlay=chunk_overlay
        self.embedding_model=embedding_model
        self.model_name=SentenceTransformer(embedding_model)
        print(f"[INFO] Loaded embedding model: {embedding_model}")
        
        
    def build_from_document(self, document:List[Any]) -> int:
        embed = Embeddings(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlay)
        chunks = embed.create_chunk(document)
        embeddings = embed.create_embeddings(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        indexed_count = len(chunks)
        print(f"[INFO] Vector store built and saved to {self.persist_dir}. Indexed {indexed_count} chunks.")
        return indexed_count
        
        
    def add_embeddings(self, embeddings:np.ndarray, metadatas:List[Any]=None):
        dim=embeddings.shape[1]
        if self.index is None:
            self.index=faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")
        
    def save(self):
        faiss_file=os.path.join(self.persist_dir, "faiss.index")
        metadata_file=os.path.join(self.persist_dir, "pickel.index")
        faiss.write_index(self.index, faiss_file)
        with open(metadata_file, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")
        
    def load(self):
        faiss_file=os.path.join(self.persist_dir, "faiss.index")
        metadata_file=os.path.join(self.persist_dir, "pickel.index")
        self.index=faiss.read_index(faiss_file)
        with open(metadata_file, "rb") as f:
            self.metadata=pickle.load(f)
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")
        
    def search(self, query_embedding:np.ndarray, top_k:int=5):
        D, I =self.index.search(query_embedding, top_k)
        results=[]
        for idx , dis in zip(I[0], D[0]):
            meta=self.metadata[idx] if idx<len(self.metadata) else None
            results.append({"index":idx, "distance":dis, "metadata":meta})
        return results
    
    def embed_query(self,query:str, top_k=5):
        print(f"[INFO] Querying vector store for: '{query}'")
        query_embed=self.model_name.encode([query]).astype('float32')
        return self.search(query_embed, top_k=top_k)
        
            
        
            
        