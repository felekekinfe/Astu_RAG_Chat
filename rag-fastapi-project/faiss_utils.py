# Placeholder
# Cell 14: Create faiss_utils.py

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List
from langchain_core.documents import Document
import os

# Initialize text splitter and embedding function
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize FAISS vector store
if os.path.exists("rag-fastapi-project/faiss_index"):
    vectorstore = FAISS.load_local("rag-fastapi-project/faiss_index", embedding_function, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_texts(["Initialize empty vector store"], embedding_function)
    vectorstore.save_local("rag-fastapi-project/faiss_index")

def load_and_split_document(file_path: str) -> List[Document]:
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    documents = loader.load()
    return text_splitter.split_documents(documents)

def index_document_to_faiss(file_path: str, file_id: int) -> bool:
    try:
        splits = load_and_split_document(file_path)
        for split in splits:
            split.metadata['file_id'] = file_id
        vectorstore.add_documents(splits)
        vectorstore.save_local("rag-fastapi-project/faiss_index")
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False

def delete_doc_from_faiss(file_id: int) -> bool:
    try:
        docs = vectorstore.get_by_ids([str(file_id)])
        if docs:
            vectorstore.delete([str(file_id)])
            vectorstore.save_local("rag-fastapi-project/faiss_index")
            print(f"Deleted documents with file_id {file_id}")
            return True
        return False
    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from FAISS: {str(e)}")
        return False