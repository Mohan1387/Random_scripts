import re

def chunk_text(text, chunk_size=500, overlap=100):
    """
    Splits a long document into overlapping chunks.

    Parameters:
    - text: Full document as a string.
    - chunk_size: Number of words per chunk.
    - overlap: Overlapping words between chunks.

    Returns:
    - List of chunked texts.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def preserve_code_blocks(text):
    """
    Ensures code blocks remain intact while chunking.
    """
    chunks = re.split(r'(```[\s\S]*?```)', text)  # Split by code blocks
    return [chunk.strip() for chunk in chunks if chunk.strip()]

# Example: Load Cybersecurity Advisory
cybersecurity_advisory_text = open("cyber_advisory.pdf", "r").read()

# Chunk while preserving code blocks
text_chunks = preserve_code_blocks(cybersecurity_advisory_text)
final_chunks = []
for chunk in text_chunks:
    final_chunks.extend(chunk_text(chunk))  # Further split long sections

print("Total Chunks:", len(final_chunks))




import ollama
import numpy as np

def get_embedding(text):
    """
    Fetches the 1024D embedding using Ollama.
    
    Parameters:
    - text: Small chunk of text.

    Returns:
    - 1024D embedding vector (NumPy array).
    """
    response = ollama.embeddings(model="bge-m3", prompt=text)
    return np.array(response["embedding"])

# Generate embeddings for each chunk
chunk_embeddings = [get_embedding(chunk) for chunk in final_chunks]




def attention_pooling(embeddings, weights=None):
    """
    Applies attention pooling over a list of embeddings.

    Parameters:
    - embeddings: List of 1024D NumPy vectors.
    - weights: Optional attention weights (defaults to equal weights).

    Returns:
    - Single 1024D vector.
    """
    embeddings = np.array(embeddings)
    num_chunks = embeddings.shape[0]

    if weights is None:
        weights = np.ones(num_chunks) / num_chunks  # Uniform weights

    weights = np.array(weights) / np.sum(weights)  # Normalize

    pooled_embedding = np.sum(weights[:, np.newaxis] * embeddings, axis=0)
    
    return pooled_embedding

# Apply attention pooling at section level
section_pooled_embeddings = [attention_pooling(chunk_embeddings[i:i+5]) for i in range(0, len(chunk_embeddings), 5)]

# Apply attention pooling at document level
final_document_embedding = attention_pooling(section_pooled_embeddings)

print("Final Advisory Embedding Shape:", final_document_embedding.shape)




import weaviate

client = weaviate.Client("http://localhost:8080")

# Define Schema
schema = {
    "class": "CybersecurityAdvisory",
    "vectorizer": "none",
    "properties": [
        {"name": "text", "dataType": ["text"]},
        {"name": "cve_id", "dataType": ["string"]},
        {"name": "severity", "dataType": ["string"]},
        {"name": "category", "dataType": ["string"]}
    ]
}
client.schema.create_class(schema)

# Store Advisory Embedding
data_object = {
    "text": cybersecurity_advisory_text, 
    "cve_id": "CVE-2024-12345",
    "severity": "Critical",
    "category": "Remote Code Execution"
}
client.data_object.create(data_object, "CybersecurityAdvisory", vector=final_document_embedding)

print("Cybersecurity Advisory Successfully Stored in Weaviate!")
