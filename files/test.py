from langchain_community.document_loaders import TextLoader


def text_loader():
    try:
        data = TextLoader("files/notes.txt")
        docs = data.load()
        return docs[0].page_content
    except Exception as e:
        return f"An error occurred: {e}"
    


