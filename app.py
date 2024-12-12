import os
import streamlit as st
from pinecone import Pinecone
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from dotenv import load_dotenv



# Load environment variables from .env file
load_dotenv()

# Set up LLM and embedding model using environment variables
llm = Gemini(api_key=os.environ["GOOGLE_API_KEY"])
embed_model = GeminiEmbedding(model_name="models/embedding-001")

# Configure settings for LLM and embeddings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1024

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Function to load documents and initialize the index in Pinecone
def ingest_documents():
    # Load documents from the specified folder
    documents = SimpleDirectoryReader("data").load_data()

    # Initialize Pinecone index and vector store
    pinecone_index = pinecone_client.Index("knowledgeagent")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # Create storage context and index from documents
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    st.success("Documents ingested successfully!")
    return index

# Streamlit App UI
st.title("Knowledge Agent Chatbot")
st.write("Ingest documents to the Pinecone index and interact with the Knowledge Agent.")

# Button to trigger document ingestion
if st.button("Ingest Documents"):
    index = ingest_documents()
    
# Create the chat interface
if 'index' in locals():
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
