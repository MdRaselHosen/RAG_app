from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("files/deeplearning.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

clean_chunks = []

for chunk in chunks:
    text = chunk.page_content
    cleaned_text = text.encode("utf-8",'ignore').decode("utf-8")

    chunk.page_content = cleaned_text
    clean_chunks.append(chunk)

    

print(type(clean_chunks))
print(type(clean_chunks[0]))
print(clean_chunks[0].page_content[:200])


embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print("Vector DB created successfully")