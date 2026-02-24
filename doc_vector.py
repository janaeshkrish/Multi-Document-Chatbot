# Note: This code is for understanding the working of RAG

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_classic.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

load_dotenv()

DIR = "/Users/janaeshwark/Desktop/langchain1.0/Multi-document-chatbot/docs"
MODEL = "gemini-2.5-flash"

llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 1. Load
loader = PyPDFDirectoryLoader(DIR)
documents = loader.load()

# 2. Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# 3. Create vector store and add documents
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embeddings,
)
vectorstore.save_local("./faiss_index")

# 4. use retrivalqa chain to answer questions
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 7}),
    return_source_documents=True,
)

result = qa_chain({"query": "When did Janaeshwar Graduate?"})
print("Answer:", result["result"])
