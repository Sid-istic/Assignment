import os
import glob
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "db_chroma")
COLLECTION_NAME = "policies"

def load_documents():
    """Load .txt files from the script's directory."""
    documents = []
    search_pattern = os.path.join(BASE_DIR, "*.txt")
    files = glob.glob(search_pattern)
    
    print(f"DEBUG: Searching for files in: {search_pattern}")
    print(f"DEBUG: Found: {files}")
    
    # Filter out system files
    valid_files = []
    for f in files:
        filename = os.path.basename(f)
        if filename not in ["requirements.txt", "LICENSE.txt", "LICENSE"]:
            valid_files.append(f)
            
    if not valid_files:
        print("WARNING: No valid .txt files found to ingest!")
        return []

    for f in valid_files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                text = file.read()
                # meaningful ID: filename without extension
                doc_name = os.path.basename(f).replace(".txt", "").replace("_", " ").title()
                # Prepend context header
                full_text = f"Source Document: {doc_name}\n\n{text}"
                documents.append({"id": f, "text": full_text, "source": doc_name})
        except Exception as e:
            print(f"Error reading {f}: {e}")

    return documents

def split_text(text):
    """Split text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    return splitter.split_text(text)

def ingest():
    """Main ingestion process."""
    print("--- Starting Ingestion ---")
    
    # 1. Load Docs
    docs = load_documents()
    if not docs:
        print("No documents to ingest.")
        return

    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection.")
    except:
        pass

    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=ef)


    ids = []
    documents_list = []
    metadatas = []

    for doc in docs:
        chunks = split_text(doc["text"])
        base_name = os.path.basename(doc["id"])
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{base_name}_{idx}"
            ids.append(chunk_id)
            documents_list.append(chunk)
            metadatas.append({"source": doc["source"]})

    if documents_list:
        print(f"Adding {len(documents_list)} chunks to database...")
        collection.add(
            ids=ids,
            documents=documents_list,
            metadatas=metadatas
        )
        print("--- Ingestion Complete ---")
    else:
        print("No chunks generated.")

if __name__ == "__main__":
    ingest()

