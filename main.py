from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from files.test import text_loader
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()


data = text_loader()

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a AI assistant that summarize the texts"),
        ("human", "{data}")
    ]
)

model = ChatMistralAI(model = "mistral-small-2603")

prompt = template.format_messages(data = data)
result = model.invoke(prompt)

print(result.content)