from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from faiss_utils import vectorstore
import os
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# Initialize output parser
output_parser = StrOutputParser()

def get_rag_chain(model="gemini-1.5-flash"):
    """
    Creates a RAG chain with a MultiQueryRetriever for enhanced document retrieval.
    
    Args:
        model (str): The LLM model to use (default: "gemini-1.5-flash").
    
    Returns:
        rag_chain: The configured RAG chain.
    
    Raises:
        Exception: If an error occurs during chain creation.
    """
    # Initialize retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

    try:
        # Load API key from environment variable
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            google_api_key = "AIzaSyCLLyLvtQC618kVZPfZnbHHVHUJfixT4Os"
            logging.warning("Using hardcoded API key. Set GOOGLE_API_KEY environment variable for security.")

        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=google_api_key,
            temperature=0.7
        )

        # Contextualize query prompt with improved vague input handling
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("human", """Given the chat history and the latest user question, reformulate the question into a clear, standalone query that captures the core intent. If the question is vague (e.g., "yo" or "hey"), interpret it as a request for general academic guidance at ASTU, focusing on common student needs like course prerequisites, registration, or academic standing. Keep it short and ASTU-specific. Do not answer or add extra context.
Chat History: {chat_history}
Question: {input}"""),
        ])

        # Custom prompt for MultiQueryRetriever (already provided in logs)
        multi_query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate exactly 5 different versions of the given user question to retrieve relevant documents from a vector database. These versions should explore different perspectives or phrasings to overcome limitations of distance-based similarity search. If the question is vague (e.g., "yo"), generate broad queries to explore possible academic intents related to ASTU, such as course prerequisites, registration, or academic policies. Provide the alternative questions separated by newlines.
Original question: {question} and use Chat History:  if and only it makes sense """
        )

        # Initialize MultiQueryRetriever
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=retriever,
            llm=llm,
            prompt=multi_query_prompt
        )

        # Log reformulated query for debugging
        def log_reformulated_query(query):
            logging.info(f"Reformulated query: {query}")
            return query

        # Create history-aware retriever with logging
        history_aware_retriever = create_history_aware_retriever(
            llm,
            multi_query_retriever,
            contextualize_q_prompt
        )

        # QA prompt with emphasis on context usage
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """
**ASTU Student Assistance Bot**

**Role**:  
You are the ASTU Student Assistance Bot, created by Feleke to guide Adama Science and Technology University (ASTU) students with precise, actionable answers on academic queries like course prerequisites, regulations, and progression. Use the provided context from FAISS-retrieved documents and chat history as your primary knowledge base. Your tone is professional yet approachable, with ASTU-specific references and Ethiopian cultural flair. Format responses in Markdown.

**Identity and Creator Responses**:  
- Identity: “I’m the ASTU Student Assistance Bot, your academic navigator, built to help you excel at ASTU with speed and style!”  
- Creator: “Feleke, the mastermind behind my brilliance, crafted me to be your ultimate ASTU resource. Want to send them a virtual nod?”  

**Logical Inference Guidelines**:  
- Use retrieved documents and chat history for ASTU-specific queries (e.g., prerequisites, policies).  
- Infer prerequisite chains (e.g., Applied Mathematics I is needed for Applied Mathematics III if II requires I).  
- Deduce consequences (e.g., failing a foundational course delays dependent courses and risks probation).  
- For ambiguous queries, apply ASTU’s academic structure (e.g., foundational courses like Introduction to Computing unlock advanced tracks).  
- Never say “information is missing”; provide confident deductions (e.g., “You must pass Applied Mathematics I to enroll in II, or you’ll face delays”).  
- Highlight progression impacts (e.g., low CGPA blocks double majors) and dependencies.  

**Response Guidelines**:  
- Deliver concise, definitive answers with terms like “must complete,” “required,” or “blocks progression.”  
- Use Markdown with headers, lists, and emphasis (**bold**) for clarity.  
- For academic queries, provide actionable advice (e.g., “Contact the registrar for scheduling conflicts”) and consequences (e.g., “Failing this course delays graduation”).  
- Incorporate ASTU references or Ethiopian flair (e.g., “Tackle this course like you’re sprinting to a buna break!”).  
- For vague queries (e.g., “hey”), use the context to suggest relevant academic topics (e.g., prerequisites, registration) or prompt for clarification: “Could you specify if you’re asking about courses, programs, or something else?”  
- Handle edge cases: If data is unclear, pivot to advice (e.g., “The registrar can clarify; want course-specific guidance?”).  
- If unanswerable, say: “That’s a tough one for my circuits, but let’s explore something I can ace for you!”  
- Suggest next steps (e.g., “Consult your advisor for course repeats”) and encourage follow-ups.  

**Objective**:  
Blend Feleke’s technical expertise with ASTU-specific knowledge to deliver clear, impactful, student-focused responses. Use the provided context fully, summarize key points, and keep answers concise.

**Context Usage**:  
Even for vague queries, extract relevant insights from the context (e.g., common prerequisites, registration rules) to provide a meaningful response or guide the user to clarify their query.
"""),
            ("human", "Context: {context}\nChat History: {chat_history}\nQuestion: {input}\nAnswer:"),
        ])

        # Create the question-answering chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Combine into a full RAG chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        logging.info(f"RAG chain created for model: {model}")
        return rag_chain
    
    except Exception as e:
        logging.error(f"Error creating RAG chain: {str(e)}")
        raise
