import os
import glob
import chromadb
from chromadb.utils import embedding_functions

# Configuration
import sys

# Get absolute path of this script's directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "db_chroma")
COLLECTION_NAME = "policies"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def load_documents():
    """Reads all .txt files from the data directory."""
    documents = []
    filenames = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    for f in filenames:
        with open(f, "r", encoding="utf-8") as file:
            documents.append({"id": f, "text": file.read()})
    return documents

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Simple recursive-like chunking: 
    Splits by paragraphs first, then combines/splits to fit chunk_size.
    """
    chunks = []
    current_chunk = ""
    
    # split by double newline first to preserve paragraphs
    paragraphs = text.split("\n\n")
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk += "\n\n" + para if current_chunk else para
        else:
            # If current chunk is full, save it
            if current_chunk:
                chunks.append(current_chunk)
            
            # If paragraph itself is huge, strictly slice it (fallback)
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i : i + chunk_size])
                current_chunk = "" # reset
            else:
                current_chunk = para # start new chunk
    
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def ingest():
    print("Loading documents...")
    docs = load_documents()
    print(f"Found {len(docs)} documents.")

    # Initialize ChromaDB
    # Using 'all-MiniLM-L6-v2' (default) which is excellent for this scale
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Reset collection if exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
        
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=ef)

    ids = []
    metadatas = []
    documents_content = []
    
    count = 0
    for doc in docs:
        source_id = doc["id"]
        chunks = split_text(doc["text"])
        
        for idx, chunk in enumerate(chunks):
            # Create a unique ID for each chunk
            ids.append(f"{os.path.basename(source_id)}_{idx}")
            metadatas.append({"source": source_id, "chunk_index": idx})
            documents_content.append(chunk)
            count += 1

    print(f"generated {count} chunks. embeddings being generated (this may take a moment)...")
    collection.add(
        documents=documents_content,
        metadatas=metadatas,
        ids=ids
    )
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest()
