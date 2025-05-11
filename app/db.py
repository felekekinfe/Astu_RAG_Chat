from chromadb import Client
from app.config import settings


def get_chroma_client():
	return Client()
	

collection=get_chroma_client.get_or_create_collection(name='astu_bot',metadata={'hnsw:space':'cosine'})


#in memory chat

chat_store={}

async def create_chat(chat_id,created):
	
	chat_store[chat_id]={'id':chat_id,'created':created, 'message':[]}
	
	return chat_store[chat_id]
	

async def add_chat_message(chat_id,message):
	
	if chat_id in chat_store:
		chat_store[chat_id]['message'].extend(message)

async def chat_exist(chat_id):
	return chat_id in chat_store

async def get_chat_message(chat_id,last_n=None):
	if chat_id in chat_store:
		messages=chat_store[chat_id]['message']
		return [{'role':m['role'],'content':m['content']} for m in messages[-last_n] if last_n else messages]
	return []
async def setup_db():
	pass
		
	
