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
    session_id = query_input.session_id or str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}, Query: {query_input.question}, Model: {query_input.model.value}")
    try:
        chat_history = get_chat_history(session_id)
        rag_chain = get_rag_chain(query_input.model.value)
        result = rag_chain.invoke({
            "input": query_input.question,
            "chat_history": chat_history
        })
        answer = result['answer']
        context = result.get('context', [])
        logging.info(f"Retrieved {len(context)} docs: {[doc.page_content[:100] for doc in context]}")
        print(f"Retrieved {len(context)} docs: {[doc.page_content[:100] for doc in context]}")
        insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
        logging.info(f"Session ID: {session_id}, Response: {answer}")
        return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)
    except Exception as e:
        logging.error(f"Error in /chat: {str(e)}")
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
