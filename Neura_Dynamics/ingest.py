# Simplified Configuration
import os
import glob
import chromadb
from chromadb.utils import embedding_functions

# Since files are flat, we look in current dir
DATA_DIR = "."
DB_DIR = "db_chroma"
COLLECTION_NAME = "policies"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

def load_documents():
    """Reads policy .txt files from the current directory."""
    documents = []
    # Match all .txt files in current directory
    filenames = glob.glob("*.txt")
    
    # Filter out known non-policy files
    ignore_list = ["requirements.txt", "LICENSE"]
    filenames = [f for f in filenames if f not in ignore_list]
    
    for f in filenames:
        with open(f, "r", encoding="utf-8") as file:
            # Prepend filename to content so LLM knows the source context
            policy_name = os.path.basename(f).replace(".txt", "").replace("_", " ").title()
            content = f"Document: {policy_name}\n\n" + file.read()
            documents.append({"id": f, "text": content})
            
    print(f"Loaded: {filenames}")
    return documents

from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Uses LangChain's RecursiveCharacterTextSplitter for robust chunking.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,    
        chunk_overlap=overlap,    
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)

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
