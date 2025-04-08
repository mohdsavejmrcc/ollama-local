import json
import os
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer

# Load a local or pre-downloaded model
model = SentenceTransformer("all-MiniLM-L6-v2")  

def generate_embedding(text):
    """Generate embeddings for a given text chunk."""
    embedding = model.encode(text, convert_to_numpy=True)  # Ensures output is a NumPy array
    return embedding

def embedding_manager(json_file_path, output_file_path="embeddings.json"):
    """Process text chunks from a JSON file and store embeddings."""
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    # Extract text chunks and ensure they exist
    text_chunks = [entry.get("chunk_text") for entry in json_data if entry.get("chunk_text")]
    
    if not text_chunks:
        print("No valid 'chunk_text' entries found in the JSON file.")
        return
    
    print(f"Found {len(text_chunks)} text chunks to process.")
    
    # Generate embeddings sequentially for debugging
    embeddings = [generate_embedding(text) for text in text_chunks]
    
    # Attach embeddings to the original data
    for entry, embedding in zip(json_data, embeddings):
        entry["embedding"] = embedding.tolist()  # Convert NumPy array to list for JSON compatibility
    
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4)
        print(f"Embeddings and metadata saved successfully to {output_file_path}!")
    except Exception as e:
        print(f"Error saving embeddings: {e}")

# Example usage
