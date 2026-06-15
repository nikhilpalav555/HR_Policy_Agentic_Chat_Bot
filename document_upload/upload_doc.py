from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from typing import List, Any


def upload_document(documents:str)->List[Any]:
    path_lib=Path(documents).resolve()
    print(f"Searching in {path_lib}")
    
    docs=[]
    
    
    pdf_files=list(path_lib.glob("**/*.pdf"))
    print(f"PDF files found: {pdf_files}")
    for pdf_file in pdf_files:
        try:
            docs_loader=PyPDFLoader(pdf_file)
            load_docs=docs_loader.load()
            docs.extend(load_docs)
            print(f"loaded {pdf_file} successfully")
        except Exception as e:
            print(f"Error loading file {e}")
            
    text_files=list(path_lib.glob("**/*.txt"))
    for text_file in text_files:
        try:
            text_load=TextLoader(text_file)
            load_text=text_load.load()
            docs.extend(load_text)
            print(f"Loaded text file successfully {text_file}")
        except Exception as e:
            print(f"Error loading file {e}")
            
    
    csv_files=list(path_lib.glob("**/*.csv"))
    for csv_file in csv_files:
        try:
            csv_load=CSVLoader(csv_file)
            load_csv=csv_load.load()
            docs.extend(load_csv)
            print(f"Loaded csv file successfully {csv_file}")
        except Exception as e:
            print(f"Error loading file {e}") 
            
    return docs
            
        
        