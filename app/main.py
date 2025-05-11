
from app.config import settings
from pdfminer.high_level import extract_text
import os

def process_pdf_docs(docs_dir=settings.Doc_Dirs):
	docs=[]
	pdf_files=[f for f in os.listdir(docs_dir) if f.lower().endwith('.pdf')]
	
	if not pdf_files:
		print('no pdf file found')
		return docs
	for filename in pdf_files:
		file_path=os.path.join(docs_dir,filename)
		
		try:
			text=extraxt_text(file_path)
			if not text.strip():
				print(f'warning: {filename} is cant be read')
				continue
			docs_name=os.path.splitext(filename)[0]
			docs.append((docs_name,text))
		except Exception as e:
			print(f'Error processing {filename}: {str(e)})
	print(f'Loaded successfully)
	return docs

def main():
	process_pdf_docs()
if __name__=='__main__':
	main()




