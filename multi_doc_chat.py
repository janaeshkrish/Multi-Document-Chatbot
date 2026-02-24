import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_classic.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain

from doc_loader import load_documents

load_dotenv()

DIR = "/Users/janaeshwark/Desktop/langchain1.0/Multi-document-chatbot/docs"
MODEL = "gemini-2.5-flash"
FAISS_INDEX_PATH = "./faiss_index"

llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=10)
vectorstore = None


def initialize_vectorstore():
    """Initialize vectorstore from saved index or create empty one"""
    global vectorstore
    if os.path.exists(FAISS_INDEX_PATH):
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings)
    else:
        documents = load_documents(DIR)
        if documents:
            docs = text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
            vectorstore.save_local(FAISS_INDEX_PATH)
        else:
            vectorstore = None
    return vectorstore


def process_and_add_documents():
    """Load documents from directory and add them to vectorstore"""
    global vectorstore

    documents = load_documents(DIR)
    if not documents:
        return False

    docs = text_splitter.split_documents(documents)
    if vectorstore is None:
        vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    else:
        vectorstore.add_documents(docs)

    vectorstore.save_local(FAISS_INDEX_PATH)
    return True


def get_qa_chain():
    """Get the conversational QA chain with current vectorstore"""
    if vectorstore is None:
        raise ValueError("No documents loaded. Please upload documents first.")

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 7}),
        return_source_documents=True,
    )


def qa_chain(inputs):
    """Execute QA chain with current vectorstore"""
    chain = get_qa_chain()
    return chain(inputs)


initialize_vectorstore()
