import streamlit as st
import weaviate
import ollama
import pypdf
import re
from weaviate.connect import ConnectionParams
import weaviate.classes.config as wc
from weaviate.collections.classes.config import DataType, Configure, Property
import weaviate.classes as wvc

# Initialize Weaviate client
client = weaviate.connect_to_local()

# Define the collection name and properties
collection_name = "Document"
properties = [
    wc.Property(name="text", data_type=wc.DataType.TEXT),
    wc.Property(name="pdf_name", data_type=wc.DataType.TEXT)
]

def setup_collection():
    """Set up the collection in Weaviate."""
    if not client.collections.exists(collection_name):
        client.collections.create(
            name=collection_name,
            properties=properties,
            vectorizer_config=[Configure.NamedVectors.text2vec_ollama(
                name="title_vector",
                source_properties=["title"],
                api_endpoint="http://host.docker.internal:11434",
                model="nomic-embed-text",
            )],
        )

setup_collection()

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    reader = pypdf.PdfReader(pdf_file)
    text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = re.split(r'(```[\s\S]*?```)', text)  # Split by code blocks
    words_chunk = [chunk.strip() for chunk in chunks if chunk.strip()]
    final_chunks = []
    for chunk in words_chunk:
        words = chunk.split()
        for i in range(0, len(words), chunk_size - overlap):
            final_chunks.append(" ".join(words[i:i + chunk_size]))
    return final_chunks

def store_in_weaviate(chunks, pdf_name):
    """Store text chunks with embeddings in Weaviate."""
    documents = client.collections.get("Document")
    documents_objects = [
        wvc.data.DataObject(properties={"text": obj, "pdf_name": pdf_name})
        for obj in chunks
    ]
    documents.data.insert_many(documents_objects)

def retrieve_similar_text(query):
    """Retrieve similar text chunks from Weaviate."""
    documents = client.collections.get("Document")
    response = documents.query.near_text(query=query, limit=2)
    return " ".join([item.properties['text'] for item in response.objects])

def query_llm(question, context, chat_history):
    """Send question and context to the LLM."""
    chat_context = "\n".join(chat_history[-5:])  # Use last 5 interactions
    prompt = f"Context: {context}\n\nChat History:\n{chat_context}\n\nUser: {question}\nAssistant:"
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Streamlit UI
st.title("CPX Threat Intel RAG Agent AI")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
user_query = st.text_input("Ask a question:")

if uploaded_file:
    if uploaded_file.name != st.session_state.uploaded_file_name:
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)
        store_in_weaviate(chunks, uploaded_file.name)
        st.session_state.uploaded_file_name = uploaded_file.name
    
    if user_query:
        context = retrieve_similar_text(user_query)
        response = query_llm(user_query, context, st.session_state.chat_history)
        st.session_state.chat_history.append(f"User: {user_query}\nAssistant: {response}")
        st.write("AI Response:", response)
else:
    if user_query:
        context = retrieve_similar_text(user_query)
        response = query_llm(user_query, context, st.session_state.chat_history)
        st.session_state.chat_history.append(f"User: {user_query}\nAssistant: {response}")
        st.write("AI Response:", response)

# Ensure Weaviate client is closed properly
import atexit
@atexit.register
def close_weaviate_client():
    client.close()
