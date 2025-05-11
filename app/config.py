from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):
	
	Allow_Origins: str='*'
	Model: str='facebook/opt-125m'
	Embedding_Model: str='sentence-transforms/all-MiniLM-L6-v2'
	Embedding_Dimensions: int=384
	Docs_Dir: str='data/docs'
	Export_Dir: str='data/export'
	Vector_Search_Top_K: int=10
	
	model_config=SettingsConfigDict(env_file='.env')
settings=Settings()
