from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from langchain_utils import get_rag_chain
from db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record
from faiss_utils import index_document_to_faiss, delete_doc_from_faiss
import os
import uuid
import logging
import shutil
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Logging setup
logging.basicConfig(filename='logs/api.log', level=logging.INFO)
app = FastAPI()

# Mount static files for the frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get('/')
async def home():
	return FileResponse('frontend/index.html')
	

# Chat endpoint
@app.post("/chat", response_model=QueryResponse)
async def chat(query_input: QueryInput):
    # Use the session_id from input if provided, otherwise generate a new one
    current_session_id = query_input.session_id 
    if not current_session_id: # This handles the first message from a new session
        current_session_id = str(uuid.uuid4())
        logging.info(f"New session created: {current_session_id}")
    else:
         logging.info(f"Using existing session: {current_session_id}")


    try:
        # --- Use the imported get_chat_history ---
        # get_chat_history from db_utils returns Langchain Message objects
        chat_history_messages = get_chat_history(current_session_id)
        # --- End Use ---
        print(f'chatttt {chat_history_messages}')
        
        # Apply history trimming (optional, based on your needs)
        # Keep the last N turns (N pairs of HumanMessage/AIMessage)
        max_history_turns = 5 # Example: keep last 5 turns (10 messages)
        if len(chat_history_messages) > max_history_turns * 2:
             # Keep only the last 'max_history_turns' pairs
            chat_history_messages = chat_history_messages[-max_history_turns * 2:]
            logging.info(f"Trimmed history for session {current_session_id} to last {max_history_turns} turns.")

        logging.info(f"Session ID: {current_session_id}, Query: {query_input.question}, Model: {query_input.model.value}")
        logging.info(f"Passing history (length {len(chat_history_messages)} messages) to RAG chain.") # Log message count

        # Make sure get_rag_chain is callable and works with your setup
        rag_chain = get_rag_chain(query_input.model.value) 

        # Invoke the RAG chain
        result = rag_chain.invoke({
            "input": query_input.question,
            "chat_history": chat_history_messages # Pass the retrieved history (Langchain Message objects)
        })

        answer = result.get('answer', 'Sorry, I could not generate an answer based on the provided information.')
        context = result.get('context', [])
        
        logging.info(f"Retrieved {len(context)} docs.")
        # You can print or log context snippet if needed, e.g., print(f"Context: {[doc.page_content[:50].replace('\\n', ' ') + '...' for doc in context]}")

        # --- Use the imported insert_application_logs to save the turn ---
        insert_application_logs(current_session_id, query_input.question, answer, query_input.model.value)
        # --- End Use ---

        logging.info(f"Session ID: {current_session_id}, Response: {answer[:100]}...") # Log snippet of response

        # Return the determined session_id in the response
        return QueryResponse(answer=answer, session_id=current_session_id, model=query_input.model)

    except Exception as e:
        logging.error(f"Error in /chat for session {current_session_id}: {str(e)}", exc_info=True) # Log exception details
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
# Upload document endpoint
@app.post("/upload-doc")
async def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = ['.pdf', '.docx', '.txt']
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {', '.join(allowed_extensions)}")
    temp_file_path = f"temp_{file.filename}"
    try:
        os.makedirs("temp", exist_ok=True)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_id = insert_document_record(file.filename)
        success = index_document_to_faiss(temp_file_path, file_id)
        if success:
            return {"message": f"File {file.filename} uploaded and indexed.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
    except Exception as e:
        logging.error(f"Error in /upload-doc: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# List documents endpoint
@app.get("/list-docs", response_model=list[DocumentInfo])
async def list_documents():
    try:
        return get_all_documents()
    except Exception as e:
        logging.error(f"Error in /list-docs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# Delete document endpoint
@app.post("/delete-doc")
async def delete_document(request: DeleteFileRequest):
    try:
        faiss_delete_success = delete_doc_from_faiss(request.file_id)
        db_delete_success = delete_document_record(request.file_id)
        if faiss_delete_success and db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id}."}
        elif faiss_delete_success:
            raise HTTPException(status_code=500, detail=f"Deleted from FAISS but failed to delete file_id {request.file_id} from database.")
        elif db_delete_success:
            raise HTTPException(status_code=500, detail=f"Deleted from database but failed to delete file_id {request.file_id} from FAISS.")
        else:
            raise HTTPException(status_code=404, detail=f"No document found with file_id {request.file_id}.")
    except Exception as e:
        logging.error(f"Error in /delete-doc: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)