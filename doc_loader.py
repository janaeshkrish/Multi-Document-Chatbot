from langchain_classic.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
import os


def load_documents(directory_path):
    documents = []
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            continue  # Skip unsupported file types
        documents.extend(loader.load())

    return documents
