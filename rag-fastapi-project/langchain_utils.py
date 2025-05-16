# Placeholder
# Cell 15: Create langchain_utils.py
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from faiss_utils import vectorstore

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
output_parser = StrOutputParser()

def get_rag_chain(model="gemini-1.5-flash"):
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key="AIzaSyCLLyLvtQC618kVZPfZnbHHVHUJfixT4Os",
        system_instruction="You are a helpful assistant that answers questions based on provided context and chat history."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("human", """
Given a chat history and the latest user question
which might reference context in the chat history,
formulate a standalone question which can be understood
without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is.
Chat History: {chat_history}
Question: {input}
        """),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("human", "Context: {context}\nChat History: {chat_history}\nQuestion: {input}\nAnswer:"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    print(f'rag chain:{rag_chain}')
    return rag_chain