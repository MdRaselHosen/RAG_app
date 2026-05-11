from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from files.pdf import pdf_loader


load_dotenv()
docs = pdf_loader()
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant that summarizes the pdf files"),
        ("human", "{data}")
    ]
)

model = ChatMistralAI(model = "mistral-small-2603")
prompt = template.format_messages(data = docs)


result = model.invoke(prompt)
print(result.content)