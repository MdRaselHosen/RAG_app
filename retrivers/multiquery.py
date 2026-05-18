from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import httpx


load_dotenv()

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent helps minimize the cost function by iteratively moving towards the steepest descent."),
    Document(page_content="Gradient descent can be used for linear regression"),
    Document(page_content="Support vector machines are a type of supervised learning algorithm used for classification and regression tasks")
]

embeddings = HuggingFaceEmbeddings()


vector_store = Chroma.from_documents(docs, embeddings)
fallback_retriever = vector_store.as_retriever()

llm = ChatMistralAI(model = "mistral-small-2603")

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=fallback_retriever,
    llm=llm,
)

query = "What is gradient descent?"

try:
    docs = multi_query_retriever.invoke(query)
except httpx.HTTPStatusError as e:
    docs = fallback_retriever.invoke(query)



print("Retrived documents-------------")

for doc in docs:
    print(doc.page_content)

