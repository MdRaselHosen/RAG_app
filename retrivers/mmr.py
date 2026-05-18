from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent helps minimize the cost function by iteratively moving towards the steepest descent."),
    Document(page_content="Gradient descent can be used for linear regression"),
    Document(page_content="Support vector machines are a type of supervised learning algorithm used for classification and regression tasks")
]

embeddings = HuggingFaceEmbeddings()

vector_store = Chroma.from_documents(docs, embeddings)

similarity_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

print("Similarity Search Results------------- ")

similarity_docs = similarity_retriever.invoke("What is gradient descent?")

for doc in similarity_docs:
    print(doc.page_content)

mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)

print("MMR search results -------------")

mmr_docs = mmr_retriever.invoke("What is gradient descent?")


for doc in mmr_docs:
    print(doc.page_content)



