import os
import streamlit as st
import multi_doc_chat
from multi_doc_chat import qa_chain, DIR, process_and_add_documents
os.makedirs(DIR, exist_ok=True)

st.set_page_config(page_title="Multi-document Chatbot", layout="wide")
st.title("Multi-document Chatbot")

with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload documents to analyze", accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("Process Documents"):
            for uploaded_file in uploaded_files:
                file_path = os.path.join(DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success(f"Successfully saved {len(uploaded_files)} files.")

            with st.spinner("Processing documents..."):
                if process_and_add_documents():
                    st.success("Documents processed and added to knowledge base!")
                else:
                    st.warning("No documents found to process.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


for msg in st.session_state.messages:  # Display chat history
    st.chat_message(msg["role"]).write(msg["content"])

# Check if documents are loaded
if multi_doc_chat.vectorstore is None:
    st.info("Please upload documents in the sidebar to get started.")
else:
    if prompt := st.chat_input("Ask questions about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    response = qa_chain(
                        {
                            "question": prompt,
                            "chat_history": st.session_state.chat_history,
                        }
                    )
                    answer = response["answer"]
                    st.write(answer)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.session_state.chat_history.append((prompt, answer))
                except Exception as e:
                    st.error(f"Error: {str(e)}")
