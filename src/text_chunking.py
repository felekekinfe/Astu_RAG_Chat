

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Text Chunking
def chunk_text(all_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text=all_text)
    print(f"Created {len(chunks)} chunks")
    return chunks
