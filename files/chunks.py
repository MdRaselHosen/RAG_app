from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader


splitter = CharacterTextSplitter(
    separator="",
    chunk_size = 10,
    chunk_overlap = 1
)

try:
    data = TextLoader("test.txt")
    docs = data.load()

    chunks = splitter.split_documents(docs)
    for i in chunks:
        print(i.page_content)
        print()
    print(len(chunks))
except Exception as e:
    print(f"An error occurred: {e}")
