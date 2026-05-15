from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence", metadata={"source": "files/deeplearning.pdf"}),
]

embedding_model = MistralAIEmbeddings()

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma_db"

)

result = vector_store.similarity_search("What is Python used for?", k=1)

for doc in result:
    print(doc.page_content)
    print(doc.metadata)

retriver = vector_store.as_retriever()
print("------------------------")

docs = retriver.invoke("Explain deep learning")

for d in docs:
    print(d.page_content)
    print(d.metadata)



