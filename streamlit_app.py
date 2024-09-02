import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import openai
import tiktoken
from tiktoken import get_encoding
import os
from dotenv import load_dotenv
import pandas as pd
from docx import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.schema import (
    SystemMessage,
    HumanMessage,
)
from langsmith import Client, trace
import functools
import re

# Load environment variables
load_dotenv()

# Access your API keys
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]

# Set environment variables for LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGCHAIN_PROJECT"] = "Gallo-Rag"

# Initialize LangSmith client
langsmith_client = Client(api_key=LANGCHAIN_API_KEY)

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "document-index"

# Initialize Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(name=INDEX_NAME, dimension=1536, metric='cosine', spec=ServerlessSpec(cloud='aws', region='us-east-1'))
index = pc.Index(INDEX_NAME)

# Define safe_run_tree decorator
def safe_run_tree(name, run_type):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with trace(name=name, run_type=run_type, client=langsmith_client) as run:
                    result = func(*args, **kwargs)
                    run.end(outputs={"result": str(result)})
                    return result
            except Exception as e:
                st.error(f"Error in LangSmith tracing: {str(e)}")
                return func(*args, **kwargs)
        return wrapper
    return decorator

def preprocess_text(text):
    # Preserve original text, only remove extra whitespace
    return ' '.join(text.split())

def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text  # Return the original text without any preprocessing

@safe_run_tree(name="generate_embedding", run_type="llm")
def generate_embedding(text):
    embeddings = OpenAIEmbeddings()
    with get_openai_callback() as cb:
        embedding = embeddings.embed_query(text.lower())
    return embedding

def upsert_document(document_text, metadata, namespace):
    max_chunk_size = 14000
    chunks = [document_text[i:i+max_chunk_size] for i in range(0, len(document_text), max_chunk_size)]
    
    for i, chunk in enumerate(chunks):
        # Generate embedding using lowercase version for better matching
        embedding = generate_embedding(chunk)
        if embedding:
            chunk_metadata = metadata.copy()
            chunk_metadata['text'] = chunk  # Store the original text with formatting
            chunk_metadata['chunk_id'] = f"{metadata['title']}_chunk_{i}"
            index.upsert(vectors=[(chunk_metadata['chunk_id'], embedding, chunk_metadata)], namespace=namespace)

def query_pinecone(query, namespace):
    query_embedding = generate_embedding(query)
    if query_embedding:
        result = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True,
            namespace=namespace
        )
        return [(match['metadata']['title'], match['metadata']['text']) for match in result['matches']]
    else:
        return []

@safe_run_tree(name="get_answer", run_type="chain")
def get_answer(context, user_query):
    chat = ChatOpenAI(model_name="gpt-4o", temperature=0.3)
    
    system_message = SystemMessage(content="""You are an AI assistant specializing in providing information based on the given context.
Your task is to answer the user's query using only the provided context related to the chosen entity.
Do not use information from other entities or sources.
Ensure your response is accurate, relevant, and concise.
Provide only the necessary answer without including the entire context.
Preserve the original formatting, capitalization, symbols like $ and %, and number representations in your answer.
Use the exact dollar amounts, percentages, and other numerical values as they appear in the context.""")
    human_message = HumanMessage(content=f"Context: {context}\n\nQuestion: {user_query}\n\nPlease provide a comprehensive answer based on the given context, preserving all original formatting, symbols, and number representations.")
    
    with get_openai_callback() as cb:
        response = chat([system_message, human_message])
    
    return response.content

@safe_run_tree(name="process_query", run_type="chain")
def process_query(query, selected_namespace):
    if query:
        with st.spinner("Searching for the best answer..."):
            matches = query_pinecone(query, selected_namespace)
            if matches:
                context = "\n\n".join([f"Title: {title}\n{text}" for title, text in matches])
                answer = get_answer(context, query)
                st.markdown(answer)  # Use markdown to preserve formatting
            else:
                st.warning("I couldn't find a specific answer to your question. Please try rephrasing or ask something else.")
    else:
        st.warning("Please enter a question before searching.")

def main():
    st.set_page_config(page_title="Document Assistant", layout="wide")

    st.title("Document Assistant")

    # Sidebar for file upload and metadata
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader("Upload DOCX Files", type="docx", accept_multiple_files=True)
        selected_namespace = st.selectbox("Select Entity (Namespace)", ["sonoma", "winery", "east coast", "farmworkers", "glass union"])
        
        if uploaded_files:
            if st.button("Upload Documents"):
                with st.spinner(f"Uploading documents to {selected_namespace} namespace..."):
                    for uploaded_file in uploaded_files:
                        document_text = extract_text_from_docx(uploaded_file)
                        metadata = {"title": uploaded_file.name}
                        upsert_document(document_text, metadata, selected_namespace)
                    st.success(f"All documents uploaded successfully to {selected_namespace} namespace!")

    # Main area for query interface
    st.header("Query Documents")
    query_namespace = st.selectbox("Select Entity to Query", ["sonoma", "winery", "east coast", "farmworkers", "glass union"])
    user_query = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        process_query(user_query, query_namespace)

if __name__ == "__main__":
    main()
