from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from faiss_utils import vectorstore  # Assuming this correctly initializes your FAISS vectorstore
import os
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO) # Keep MultiQuery logging on

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
            # NOTE: Using a hardcoded key like this is HIGHLY INSECURE.
            # This is just to match the user's original code's behavior for reproducibility.
            # ALWAYS set the GOOGLE_API_KEY environment variable in a real application.
            google_api_key = "AIzaSyCLLyLvtQC618kVZPfZnbHHVHUJfixT4Os" 
            logging.warning("Using hardcoded API key. Set GOOGLE_API_KEY environment variable for security.")

        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=google_api_key,
            temperature=0.4
        )

        # Contextualize query prompt
        # This prompt takes chat history and the latest input to create a clear, standalone query.
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("human", """Given the chat history and the latest user question, reformulate the question into a clear, standalone query that captures the core intent. This query will be used to retrieve relevant documents.
If the question is vague (e.g., "yo" or "hey"), interpret it as a greeting request for general academic guidance at ASTU, focusing on common student needs like course prerequisites, registration, or academic standing, and formulate a query to find general ASTU academic info.
Keep the standalone query short and ASTU-specific. Do not answer the question or add extra context like 'Based on...', etc.
Chat History: {chat_history}
Question: {input}"""),
        ])

        # --- Inject doc.txt content into MultiQuery Prompt Template ---
        multi_query_context_string = ""
        try:
            with open('code.txt', 'r') as f:
                multi_query_context_string = f.read()
            logging.info("Successfully read doc.txt for multi-query context.")
        except FileNotFoundError:
            logging.warning("doc.txt not found. Multi-query generation will not use specific context from this file.")
        except Exception as e:
            logging.error(f"Error reading doc.txt for multi-query context: {e}")

        # Construct the template string with the doc.txt content embedded
        # The LLM is instructed to use this embedded context when generating queries.
        multi_query_template_string = f"""You are an AI language model assistant. Your task is to generate exactly 5 different versions of the given user question to retrieve relevant documents from a vector database. These versions should explore different perspectives or phrasings to overcome limitations of distance-based similarity search.
        if its greeting g hi or other give greeting back politely
        If the question is vague (e.g., "yo", "hey", "tell me about this"), use the provided ASTU academic context below to generate broad queries related to common student inquiries (e.g., course prerequisites, registration, academic policies) that are relevant to the context.
	If the question is like 'hi','hello' 'whatsup'..etc kind of greeting use greting dont generate query about astu.
        If the question is specific, generate 5 variations that rephrase or explore related aspects of the specific question, also keeping the ASTU context in mind.

        Provide the alternative questions separated by newlines.

        Original question: {{question}}

        ASTU Academic Context (Use this to guide your query variations, especially for vague questions):
        {multi_query_context_string}

        """
        # --- End Injection ---

        # Now create the PromptTemplate for MultiQueryRetriever
        # It still only needs 'question' as an input variable because the doc.txt content
        # is now part of the static template string itself.
        multi_query_prompt = PromptTemplate(
            input_variables=["question"], 
            template=multi_query_template_string
        )

        # Initialize MultiQueryRetriever with the corrected prompt
        # This retriever will receive the *standalone query* from the history-aware step
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=retriever,
            llm=llm,
            prompt=multi_query_prompt # Use the corrected prompt
        )

        # Create history-aware retriever
        # This chain takes the raw input and chat history, uses contextualize_q_prompt
        # to generate a standalone query, and then passes that standalone query
        # to the underlying retriever (multi_query_retriever).
        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=multi_query_retriever, # The multi_query_retriever is the underlying retriever
            prompt=contextualize_q_prompt
        )

        # QA prompt with emphasis on context usage and explicit source avoidance
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """
        **ASTU Student Assistance Bot**

        **Role**:
        You are the ASTU Student Assistance Bot, created by Feleke to guide Adama Science and Technology University (ASTU) students with precise, actionable answers on academic queries like course prerequisites, regulations, and progression. Use the provided context from FAISS-retrieved documents and chat history as your primary knowledge base. Your tone is professional yet approachable, with ASTU-specific references and Ethiopian cultural flair. Format responses in Markdown.

        **Identity and Creator Responses**:
        - Identity: “I’m the ASTU Student Assistance Bot, your academic navigator, built to help you excel at ASTU!”
        - Creator: “Feleke, the mastermind behind my brilliance, crafted me to be your ultimate ASTU resource. Want to send them a virtual nod?”

        **Logical Inference Guidelines**:
        - Use retrieved documents and chat history for ASTU-specific queries (e.g., prerequisites, policies).
        - Infer prerequisite chains (e.g., Applied Mathematics I is needed for Applied Mathematics III if II requires I).
        - Deduce consequences (e.g., failing a foundational course delays dependent courses and risks probation).
        - For ambiguous queries, apply ASTU’s academic structure (e.g., foundational courses like Introduction to Computing unlock advanced tracks).
        - Never say “information is missing”; provide confident deductions (e.g., “You must pass Applied Mathematics I to enroll in II, or you’ll face delays”).
        - Highlight progression impacts (e.g., low CGPA blocks double majors) and dependencies.

        **Response Guidelines**:
        - **CRITICAL: ABSOLUTELY DO NOT REFERENCE THE SOURCE OR PROVIDED CONTEXT IN YOUR RESPONSE.** Never use phrases like "Based on the provided text," "According to the documents," "From the context," "The documents say," "The provided information does not mention," "According to the context," or *any* similar language that refers to the origin of the information (or the lack of information).
        - Deliver concise, definitive answers with terms like “must complete,” “required,” or “blocks progression.”
        - Use Markdown with headers, lists, and emphasis (**bold**) for clarity.
        - For academic queries, provide actionable advice (e.g., “Contact the registrar for scheduling conflicts”) and consequences (e.g., “Failing this course delays graduation”).
        - Incorporate ASTU references or Ethiopian flair (e.g., “Tackle this course like you’re sprinting to a buna break!”).
        - For vague queries (e.g., “hey”), use the context to suggest relevant academic topics (e.g., common prerequisites, registration rules) or prompt for clarification based on common student needs: “Could you specify if you’re asking about courses, programs, or something else related to ASTU academics?”
        - **Handle edge cases/unanswerable questions:** If you cannot find the answer based on the provided context, state the fallback phrase: “That’s a tough one for my circuits, but let’s explore something I can ace for you!” **Do not add any further explanation about why you couldn't find the answer or mention that the information was not in the documents/context.** Then, suggest an alternative action if applicable (like checking the ASTU website or consulting an advisor).
        - Suggest next steps (e.g., “Consult your advisor for course repeats”) and encourage follow-ups.

        **Objective**:
        Blend Feleke’s technical expertise with ASTU-specific knowledge to deliver clear, impactful, student-focused responses. Use the provided context fully, summarize key points, and keep answers concise.

        **Context Usage**:
        Even for vague queries, extract relevant insights from the context (e.g., common prerequisites, registration rules) to provide a meaningful response or guide the user to clarify their query.
        """),
            # The QA prompt receives the original input, chat history, AND the retrieved context
            ("human", "Context: {context}\nChat History: {chat_history}\nQuestion: {input}\nAnswer:"),
        ])

        # Create the question-answering chain which combines the retrieved documents with the input
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Combine into a full RAG chain
        # This chain orchestrates the process:
        # 1. Takes chat_history and input.
        # 2. Uses history_aware_retriever (which internally uses contextualize_q_prompt)
        #    to generate a standalone query.
        # 3. Passes the standalone query to the underlying retriever (multi_query_retriever)
        #    to get documents.
        # 4. Passes the original input, chat_history, and retrieved documents to the
        #    question_answer_chain (using qa_prompt) to generate the final answer.
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        logging.info(f"RAG chain created for model: {model}")
        return rag_chain
    
    except Exception as e:
        logging.error(f"Error creating RAG chain: {str(e)}")
        raise

# Example Usage (requires a vectorstore initialized in faiss_utils.py)
if __name__ == '__main__':
    try:
        # Initialize the RAG chain
        rag_chain = get_rag_chain()

        # Example conversation
        chat_history = []

        print("ASTU Student Assistance Bot (Type 'exit' to quit)")

        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break

            try:
                # Invoke the RAG chain
                response = rag_chain.invoke({
                    "chat_history": chat_history,
                    "input": user_input
                })

                # Process the response (assuming it has an 'answer' key and potentially 'context' or 'source_documents')
                # The structure depends on the chain type. create_retrieval_chain returns {"answer": ..., "context": ...}
                bot_response = response.get("answer", "Sorry, I couldn't process that.")
                # retrieved_docs = response.get("context", []) # You can inspect retrieved docs if needed
                # logging.info(f"Retrieved documents count: {len(retrieved_docs)}") # Optional: Log retrieved docs count

                print(f"Bot: {bot_response}")

                # Update chat history for the next turn
                chat_history.extend([
                    ("human", user_input),
                    ("ai", bot_response)
                ])

            except Exception as e:
                logging.error(f"Error during conversation turn: {e}")
                print("Bot: An error occurred while processing your request. Please try again.")

    except Exception as e:
        logging.error(f"Failed to initialize RAG chain: {e}")
        print("Error initializing the chatbot. Please check logs.")
