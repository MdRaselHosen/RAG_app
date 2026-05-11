from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 10
)
def pdf_loader():
    try:

        data = PyPDFLoader("notes.pdf")

        docs = data.load()

        return docs
    
    except Exception as e:
        return f"An error occurred: {e}"
    
data = pdf_loader()
chunks = splitter.split_documents(data)
print(len(chunks))
print(chunks[0].page_content)

