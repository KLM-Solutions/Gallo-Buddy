import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import tiktoken
from tiktoken import get_encoding
import os
from dotenv import load_dotenv
import pandas as pd
from docx import Document
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.schema import (
    SystemMessage,
    HumanMessage,
)
from langsmith import Client, trace
import functools

# Load environment variables
load_dotenv()

# Access your API keys
# Load API keys and set environment variables
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
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "document-index"

# Initialize Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(name=INDEX_NAME, dimension=1536, metric='cosine', spec=ServerlessSpec(cloud='aws', region='us-east-1'))
index = pc.Index(INDEX_NAME)

# Initialize OpenAIEmbeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

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

def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

@safe_run_tree(name="generate_embedding", run_type="llm")
def generate_embedding(text):
    with get_openai_callback() as cb:
        embedding = embeddings.embed_query(text)
    return embedding

@safe_run_tree(name="upsert_document", run_type="chain")
def upsert_document(document_text, metadata, namespace):
    max_chunk_size = 14000
    chunks = [document_text[i:i+max_chunk_size] for i in range(0, len(document_text), max_chunk_size)]
    
    for i, chunk in enumerate(chunks):
        with trace(name=f"process_chunk_{i}", run_type="chain", client=langsmith_client) as chunk_run:
            embedding = generate_embedding(chunk)
            if embedding:
                chunk_metadata = metadata.copy()
                chunk_metadata['text'] = chunk
                chunk_metadata['chunk_id'] = f"{metadata['title']}_chunk_{i}"
                index.upsert(vectors=[(chunk_metadata['chunk_id'], embedding, chunk_metadata)], namespace=namespace)
            chunk_run.end(outputs={"chunk_id": chunk_metadata['chunk_id']})

@safe_run_tree(name="query_pinecone", run_type="chain")
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

def truncate_context(context, max_tokens=3000):
    encoding = tiktoken.encoding_for_model("gpt-4o")
    encoded_context = encoding.encode(context)
    
    if len(encoded_context) > max_tokens:
        truncated_context = encoding.decode(encoded_context[:max_tokens])
        return truncated_context
    return context

@safe_run_tree(name="get_answer", run_type="chain")
def get_answer(context, user_query):
    chat = ChatOpenAI(model_name="gpt-4o", temperature=0.3, openai_api_key=OPENAI_API_KEY)
    
    # Truncate the context before passing it to the LLM
    truncated_context = truncate_context(context)
    
    system_message = SystemMessage(content="""You are an AI assistant specializing in providing information based on the given context.
Your task is to answer the user's query using only the provided context related to the chosen entity: {chosen_entity}.
Do not use information from other entities or sources.
Ensure your response is accurate, relevant, and concise.
Provide only the necessary answer without including the entire context.""")
    human_message = HumanMessage(content=f"Context: {truncated_context}\n\nQuestion: {user_query}\n\nPlease provide a comprehensive answer based on the given context.")
    
    with get_openai_callback() as cb:
        response = chat([system_message, human_message])
    
    return response.content

@safe_run_tree(name="process_query", run_type="chain")
def process_query(query, selected_namespace):
    if query:
        with st.spinner("Searching for the best answer..."):
            with trace(name="process_query_detail", run_type="chain", client=langsmith_client) as run:
                # Add input to the trace
                run.inputs["query"] = query
                run.inputs["namespace"] = selected_namespace
                
                matches = query_pinecone(query, selected_namespace)
                if matches:
                    context = " ".join([f"Title: {title}\n{text}" for title, text in matches])
                    answer = get_answer(context, query)
                    st.write(answer)
                    
                    # Add both query and answer to the trace outputs
                    run.outputs["query"] = query
                    run.outputs["answer"] = answer
                else:
                    st.warning("I couldn't find a specific answer to your question. Please try rephrasing or ask something else.")
                    
                    # Add query and the "no answer" response to the trace outputs
                    run.outputs["query"] = query
                    run.outputs["answer"] = "No specific answer found"
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
