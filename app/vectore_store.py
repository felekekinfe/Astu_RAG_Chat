
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


# Vector Store
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store


# Format Retrieved Documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)