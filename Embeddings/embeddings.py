from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Any
import numpy as np


class Embeddings:
    def __init__(self, model_name:str="all-MiniLM-L6-v2", chunk_size=1000, chunk_overlap=200)->None:
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.model=SentenceTransformer(model_name)
        print(f"loading model {self.model}")
        
    
    def create_chunk(self,documents:List[Any])->List[Any]:
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks=text_splitter.split_documents(documents)
        return chunks
    
    def create_embeddings(self,chunks:List[Any])->np.ndarray:
        texts=[chunk.page_content for chunk in chunks]
        embeddings=self.model.encode(texts, show_progress_bar=True)
        print(f"Embeddings shape {embeddings.shape}")
        return embeddings
        