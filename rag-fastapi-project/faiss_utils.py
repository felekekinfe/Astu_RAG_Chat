from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List
from langchain_core.documents import Document
import os

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

FAISS_INDEX_PATH = "faiss_index/"
if os.path.exists(FAISS_INDEX_PATH):
    try:
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding_function, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading FAISS index: {str(e)}")
        vectorstore = FAISS.from_texts(["Initialize empty vector store"], embedding_function)
else:
    vectorstore = FAISS.from_texts(["Initialize empty vector store"], embedding_function)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"Created new FAISS index at {FAISS_INDEX_PATH}")

def load_and_split_document(file_path: str) -> List[Document]:
    print(f"Loading document: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        documents = loader.load()
        print(f"Loaded {len(documents)} raw documents")
        splits = text_splitter.split_documents(documents)
        print(f"Split into {len(splits)} document chunks")
        return splits
    except Exception as e:
        print(f"Error loading document {file_path}: {str(e)}")
        return []

def index_document_to_faiss(file_path: str, file_id: int) -> bool:
    try:
        splits = load_and_split_document(file_path)
        if not splits:
            print(f"No document splits created for {file_path}")
            return False
        for split in splits:
            split.metadata['file_id'] = file_id
        vectorstore.add_documents(splits)
        vectorstore.save_local(FAISS_INDEX_PATH)
        clean_placeholder_document()
        print(f"Successfully indexed {len(splits)} splits for file_id {file_id}")
        return True
    except Exception as e:
        print(f"Error indexing document {file_path}: {str(e)}")
        return False

def delete_doc_from_faiss(file_id: int) -> bool:
    try:
        all_docs = vectorstore.docstore._dict
        ids_to_delete = [
            doc_id for doc_id, doc in all_docs.items()
            if doc.metadata.get('file_id') == file_id
        ]
        if ids_to_delete:
            vectorstore.delete(ids_to_delete)
            vectorstore.save_local(FAISS_INDEX_PATH)
            print(f"Deleted {len(ids_to_delete)} document splits with file_id {file_id}")
            return True
        else:
            print(f"No documents found with file_id {file_id}")
            return False
    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from FAISS: {str(e)}")
        return False

def clean_placeholder_document():
    try:
        all_docs = vectorstore.docstore._dict
        ids_to_delete = [
            doc_id for doc_id, doc in all_docs.items()
            if doc.page_content == "Initialize empty vector store" and not doc.metadata.get('file_id')
        ]
        if ids_to_delete:
            vectorstore.delete(ids_to_delete)
            vectorstore.save_local(FAISS_INDEX_PATH)
            print(f"Deleted {len(ids_to_delete)} placeholder documents")
    except Exception as e:
        print(f"Error cleaning placeholder document: {str(e)}")

def test_retriever(query: str):
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)
        docs = [doc for doc in docs if doc.page_content != "Initialize empty vector store"]
        print(f"Retrieved {len(docs)} documents for query '{query}':")
        for doc in docs:
            print(f" - {doc.page_content[:100]}... (file_id: {doc.metadata.get('file_id')})")
        return docs
    except Exception as e:
        print(f"Error in retriever: {str(e)}")
        return []