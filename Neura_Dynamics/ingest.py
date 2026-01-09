import os
import glob
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Get absolute directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "db_chroma")
COLLECTION_NAME = "policies"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

def load_documents():
    documents = []
    # Use absolute path to find .txt files
    search_path = os.path.join(BASE_DIR, "*.txt")
    filenames = glob.glob(search_path)
    
    print(f"DEBUG: Searching in: {search_path}")
    print(f"DEBUG: Found files: {filenames}")
    
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

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


def ingest():
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
    print("\n[3/3] Chunking documents...")
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
    
    
    collection.add(
        documents=documents_content,
        metadatas=metadatas,
        ids=ids
    )
    
    print("✓ Embeddings generated and stored")
    print("\n" + "=" * 60)
    print("✅ INGESTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    ingest()
