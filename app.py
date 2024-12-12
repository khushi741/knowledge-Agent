import streamlit as st
from pinecone import Pinecone
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings

# Streamlit App UI
st.title("Knowledge Agent Chatbot")
st.write("Ingest documents to the Pinecone index and interact with the Knowledge Agent.")

# Input for API keys
google_api_key = st.text_input("Enter your Google Gemini API Key:", type="password")
pinecone_api_key = st.text_input("Enter your Pinecone API Key:", type="password")

if google_api_key and pinecone_api_key:
    try:
        # Initialize Gemini LLM and embedding model
        llm = Gemini(api_key=google_api_key)
        embed_model = GeminiEmbedding(model_name="models/embedding-001")

        # Configure settings for LLM and embeddings
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = 1024

        # Initialize Pinecone client
        pinecone_client = Pinecone(api_key=pinecone_api_key)

        st.success("API keys validated successfully!")
    except Exception as e:
        st.error(f"Error initializing services: {str(e)}")
else:
    st.warning("Please provide both API keys to proceed.")

# Global variable to store the index
index = None

# Function to load documents and initialize the index in Pinecone
def ingest_documents():
    global index
    try:
        # Load documents from the specified folder
        documents = SimpleDirectoryReader("data").load_data()

        # Initialize Pinecone index and vector store
        pinecone_index = pinecone_client.Index("knowledgeagent")
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

        # Create storage context and index from documents
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        st.success("Documents ingested successfully!")
    except Exception as e:
        st.error(f"Error during document ingestion: {str(e)}")

# Button to trigger document ingestion
if google_api_key and pinecone_api_key:
    if st.button("Ingest Documents"):
        ingest_documents()
else:
    st.warning("Provide API keys first to enable document ingestion.")

# Chat interface
if google_api_key and pinecone_api_key:
    if index:
        try:
            # Set up the chat engine after index is created
            chat_engine = index.as_chat_engine()

            # Create input box for user to type query
            user_input = st.text_input("You: ", "")

            if user_input:
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

    

           
