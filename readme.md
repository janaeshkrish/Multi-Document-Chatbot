# Multi-Document Chatbot

A Retrieval-Augmented Generation (RAG) application that enables users to upload multiple documents and interact with their content through a conversational interface. 

![Chatbot Screenshot](screenshot.png)

## Features

- **Multi-Document Support**: Upload and process multiple files (PDF, DOCX, TXT) simultaneously via a streamlined sidebar.
- **Contextual Retrieval**: Uses vector embeddings to find the most relevant information from your documents to answer queries accurately.
- **Conversational Memory**: Maintains chat history to contextualize questions across multiple turns.
- **Streamlined UI**: Interactive, dark-mode ready chat interface built with Streamlit, featuring a dedicated document management space.
- **On-Demand Processing**: Explicitly process documents using the "Process Documents" button when you're ready, saving compute and avoiding unwanted auto-refreshes.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Multi-Document-Chatbot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup**:
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```
2. **Upload Documents**: Use the "Document Upload" sidebar to drag-and-drop or browse your PDFs, text, or word documents.
3. **Process**: Click the "Process Documents" button to extract text, generate embeddings, and construct the FAISS vector database.
4. **Chat**: Once processed, utilize the bottom chat input ("Ask questions about your documents") to extract insights, compare information across resumes/documents, or summarize content!

## Tech Stack

- **LLM Orchestration**: LangChain, Google Gemini (`gemini-2.5-flash`)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Frontend**: Streamlit
- **Embeddings**: Google Generative AI Embeddings (`models/gemini-embedding-001`)
