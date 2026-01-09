"""
Data Ingestion Pipeline for RAG System
Loads policy documents, chunks them, and stores in ChromaDB vector database.
"""

import os
import glob
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================================
# CONFIGURATION
# ============================================================================

DB_DIR = "db_chroma"
COLLECTION_NAME = "policies"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# ============================================================================
# DOCUMENT LOADING
# ============================================================================

def load_documents():
    """
    Loads all policy .txt files from the current directory.
    
    Returns:
        list: List of dicts with 'id' and 'text' keys
    """
    documents = []
    
    # Find all .txt files in current directory
    filenames = glob.glob("*.txt")
    
    # Debug output
    print(f"DEBUG: Current working directory: {os.getcwd()}")
    print(f"DEBUG: Found .txt files: {filenames}")
    
    # Early return if no files found
    if not filenames:
        print("ERROR: No .txt files found to ingest!")
        return []
    
    # Filter out non-policy files
    ignore_list = ["requirements.txt", "LICENSE.txt"]
    filenames = [f for f in filenames if f not in ignore_list]
    
    # Load each file
    for filename in filenames:
        with open(filename, "r", encoding="utf-8") as file:
            # Extract clean policy name from filename
            policy_name = os.path.basename(filename).replace(".txt", "").replace("_", " ").title()
            
            # Prepend document identifier to help LLM understand context
            content = f"Document: {policy_name}\n\n{file.read()}"
            
            documents.append({
                "id": filename,
                "text": content
            })
    
    print(f"Successfully loaded: {filenames}")
    return documents

# ============================================================================
# TEXT CHUNKING
# ============================================================================

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Splits text into smaller chunks using LangChain's RecursiveCharacterTextSplitter.
    
    Args:
        text (str): Text to split
        chunk_size (int): Maximum characters per chunk
        overlap (int): Number of overlapping characters between chunks
        
    Returns:
        list: List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)

# ============================================================================
# MAIN INGESTION FUNCTION
# ============================================================================

def ingest():
    """
    Main ingestion pipeline:
    1. Load documents from disk
    2. Split into chunks
    3. Generate embeddings
    4. Store in ChromaDB
    """
    print("=" * 60)
    print("STARTING DATA INGESTION")
    print("=" * 60)
    
    # Step 1: Load documents
    print("\n[1/4] Loading documents...")
    docs = load_documents()
    
    if not docs:
        print("❌ No documents loaded. Aborting ingestion.")
        return
    
    print(f"✓ Found {len(docs)} documents")
    
    # Step 2: Initialize ChromaDB
    print("\n[2/4] Initializing ChromaDB...")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Reset collection if it exists (ensures clean state)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"✓ Deleted existing collection '{COLLECTION_NAME}'")
    except:
        print(f"✓ No existing collection to delete")
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef
    )
    print(f"✓ Created collection '{COLLECTION_NAME}'")
    
    # Step 3: Chunk documents
    print("\n[3/4] Chunking documents...")
    ids = []
    metadatas = []
    documents_content = []
    
    for doc in docs:
        source_id = doc["id"]
        chunks = split_text(doc["text"])
        
        for idx, chunk in enumerate(chunks):
            # Create unique ID for each chunk
            chunk_id = f"{os.path.basename(source_id)}_{idx}"
            ids.append(chunk_id)
            
            # Store metadata for traceability
            metadatas.append({
                "source": source_id,
                "chunk_index": idx
            })
            
            documents_content.append(chunk)
    
    print(f"✓ Generated {len(documents_content)} chunks")
    
    # Step 4: Add to database (embeddings generated here)
    print("\n[4/4] Generating embeddings and storing in database...")
    print("(This may take a moment...)")
    
    collection.add(
        documents=documents_content,
        metadatas=metadatas,
        ids=ids
    )
    
    print("✓ Embeddings generated and stored")
    print("\n" + "=" * 60)
    print("✅ INGESTION COMPLETE")
    print("=" * 60)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    ingest()
