from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from files.pdf import pdf_loader


load_dotenv()

docs = pdf_loader()

spiller = RecursiveCharacterTextSplitter(
    chunk_size= 1000,
    chunk_overlap = 200
)

chunks = spiller.split_documents(docs)

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant that summarizes the pdf files"),
        ("human", "{data}")
    ]
)

model = ChatMistralAI(model = "mistral-small-2603")
prompt = template.format_messages(data = chunks[0].page_content)


result = model.invoke(prompt)
print(result.content)