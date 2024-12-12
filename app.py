import os
import streamlit as st
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Function to set up the document ingestion process
def ingest_documents():
    if not os.path.exists("data") or not os.listdir("data"):
        st.error("The 'data' directory is empty or does not exist. Please upload documents.")
        return None

    # Load documents from the 'data' directory
    documents = SimpleDirectoryReader("data").load_data()

    # Initialize Pinecone index and vector store
    pinecone_index = pinecone_client.Index("knowledgeagent")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # Create storage context and index from documents
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    st.success("Documents ingested successfully!")
    return index

# Function to handle file uploads
def handle_upload(uploaded_file):
    if uploaded_file is not None:
        if not os.path.exists("data"):
            os.makedirs("data")

        # Save the uploaded file to the 'data' directory
        with open(f"data/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Uploaded file: {uploaded_file.name}")
    else:
        st.error("No file uploaded.")

# Streamlit App UI
st.title("Knowledge Agent Chatbot")
st.write("Ingest documents to the Pinecone index and interact with the Knowledge Agent.")

# File uploader for dynamic uploads
uploaded_file = st.file_uploader("Upload a document to ingest", type=["txt", "pdf", "docx"])
if uploaded_file:
    handle_upload(uploaded_file)

# Button to trigger document ingestion
index = None
if st.button("Ingest Documents"):
    index = ingest_documents()

# Create the chat interface
if index:
    # Set up the chat engine after index is created
    chat_engine = index.as_chat_engine()

    # Create input box for user to type query
    user_input = st.text_input("You: ", "")

    if user_input:
        try:
            # Get response from the chat engine
            response = chat_engine.chat(user_input)

            # Display agent's response
            st.write(f"Agent: {response.response}")
        except Exception as e:
            st.error(f"Error during query: {str(e)}")
else:
    st.warning("Please ingest documents first by clicking the 'Ingest Documents' button.")

# Optional: Clear the session state to reset the app
if st.button("Clear"):
    st.experimental_rerun()
