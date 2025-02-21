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


def format_prompt(query, retrieved_advisories):
    """
    Formats retrieved advisory data into a structured prompt for the LLM.
    """
    context = "\n".join([
        f"CVE ID: {advisory['cve_id']}\nMitigation Steps: {', '.join(advisory['mitigation_steps'])}"
        for advisory in retrieved_advisories
    ])

    prompt = f"""
    You are a cybersecurity expert. Below are the details of relevant security advisories retrieved from a database.
    
    Context:
    {context}
    
    User Query:
    {query}
    
    Based on the provided context, provide a clear and actionable response.
    """

    return prompt

# Create a formatted prompt
formatted_prompt = format_prompt(query, retrieved_advisories)

print(formatted_prompt)  # Debugging: See the final prompt being sent to the LLM


import ollama

def query_llm(prompt, model="llama3"):
    """
    Sends a prompt to an Ollama-hosted LLM and returns the generated response.
    
    Parameters:
    - prompt: Formatted input text including retrieved cybersecurity advisories.
    - model: The LLM to use (default: llama3).
    
    Returns:
    - LLM-generated response.
    """
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Get final answer from LLM
final_answer = query_llm(formatted_prompt)

print("Generated Response from LLM:\n", final_answer)




query = client.query.get("CybersecurityAdvisory", ["cve_id", "risk_metadata"]) \
    .with_where({
        "path": ["risk_metadata", "cvss_score"],
        "operator": "GreaterThan",
        "valueNumber": 9.0
    }) \
    .with_limit(5) \
    .do()

print(query["data"]["Get"]["CybersecurityAdvisory"])






import weaviate

client = weaviate.Client("http://localhost:8080")

# Define Schema with Dictionary, List, String, and Vector
schema = {
    "class": "CybersecurityAdvisory",
    "vectorizer": "none",  # Using custom embeddings
    "properties": [
        {"name": "cve_id", "dataType": ["string"]},  # CVE identifier
        {"name": "attack_techniques", "dataType": ["string"]},  # List of attack methods
        {"name": "affected_systems", "dataType": ["string"]},  # List of OS/software affected
        {"name": "risk_metadata", "dataType": ["object"]},  # Dictionary with risk details
        {"name": "mitigation_steps", "dataType": ["string"]},  # List of recommended fixes
    ]
}

client.schema.create_class(schema)

print("Schema Created Successfully!")




import ollama
import numpy as np

def get_embedding(text):
    """Fetches the 1024D embedding using Ollama."""
    response = ollama.embeddings(model="bge-m3", prompt=text)
    return np.array(response["embedding"])

# Define Advisory Object
advisory_data = {
    "cve_id": "CVE-2024-12345",
    "attack_techniques": ["Remote Code Execution", "Privilege Escalation"],
    "affected_systems": ["Linux", "Windows Server", "macOS"],
    "risk_metadata": {  # Nested dictionary
        "cvss_score": 9.8,
        "exploitable": True,
        "disclosure_date": "2024-06-01",
        "exploit_references": [
            "https://nvd.nist.gov/vuln/detail/CVE-2024-12345",
            "https://exploit-db.com/exploits/12345"
        ]
    },
    "mitigation_steps": [
        "Apply the latest OpenSSH patch",
        "Disable password authentication in SSH config",
        "Monitor system logs for suspicious activity"
    ]
}

# Generate vector embedding for the advisory
embedding = get_embedding(str(advisory_data))  # Convert dict to string for embedding

# Store the advisory with vector
client.data_object.create(advisory_data, "CybersecurityAdvisory", vector=embedding)

print("Cybersecurity Advisory Successfully Stored!")

