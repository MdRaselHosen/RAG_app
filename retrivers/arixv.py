from langchain_community.retrievers import ArxivRetriever

retriver = ArxivRetriever(
    load_max_docs=2,
    load_all_available_meta=True
)

docs = retriver.invoke("large language models")

for i,doc in enumerate(docs):
    print(f"Document {i+1}:")
    print("Title:", doc.metadata.get("title"))
    print("Authors:", doc.metadata.get("authors"))
    print("Abstract:", doc.page_content[:200], "...")
    print()

