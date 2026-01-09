"""
RAG (Retrieval-Augmented Generation) Module
Handles document retrieval and answer generation using local LLM.
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

DB_DIR = "db_chroma"
COLLECTION_NAME = "policies"
TOP_K = 3  # Number of relevant chunks to retrieve
MODEL_NAME = "google/flan-t5-base"

# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end. Answer in detail and complete sentences. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE
)

# ============================================================================
# DATABASE SETUP
# ============================================================================

# Initialize embedding function
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Connect to ChromaDB
client = chromadb.PersistentClient(path=DB_DIR)

def get_collection():
    """
    Gets or creates the collection. Called dynamically to handle rebuilds.
    """
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef
    )

def get_doc_count():
    """
    Returns the number of documents in the collection.
    Used by app.py to check if database needs initialization.
    """
    try:
        collection = get_collection()
        return collection.count()
    except Exception as e:
        # If collection doesn't exist or is corrupted, return 0
        print(f"Warning: Could not get doc count: {e}")
        return 0

# ============================================================================
# MODEL LOADING
# ============================================================================

# Smart caching: Use Streamlit's cache if available, otherwise use dummy decorator
try:
    import streamlit as st
    cache_decorator = st.cache_resource
except ImportError:
    # Dummy decorator for CLI usage
    def cache_decorator(func):
        return func

@cache_decorator
def load_model():
    """
    Loads the Flan-T5 model and creates a text generation pipeline.
    Cached to avoid reloading on every request.
    """
    print(f"Loading local model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512
    )

# Initialize the pipeline
pipe = load_model()

# ============================================================================
# RETRIEVAL FUNCTION
# ============================================================================

def retrieve_context(query):
    """
    Retrieves the most relevant document chunks for a given query.
    
    Args:
        query (str): User's question
        
    Returns:
        str: Concatenated relevant chunks, or empty string if none found
    """
    try:
        collection = get_collection()
        
        # Safety check: Return empty if database is empty
        if collection.count() == 0:
            return ""
        
        # Query the vector database
        results = collection.query(
            query_texts=[query],
            n_results=TOP_K
        )
        
        # Safety check: Ensure results exist
        if not results['documents'] or not results['documents'][0]:
            return ""
        
        # Extract and concatenate document chunks
        docs = results['documents'][0]
        return "\n---\n".join(docs)
    
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return ""

# ============================================================================
# ANSWER GENERATION
# ============================================================================

def generate_answer(query):
    """
    Main RAG function: Retrieves context and generates answer.
    
    Args:
        query (str): User's question
        
    Returns:
        str: Generated answer or error message
    """
    # Step 1: Retrieve relevant context
    context = retrieve_context(query)
    
    # Step 2: Format prompt with context and question
    formatted_prompt = prompt_template.format(
        context=context,
        question=query
    )
    
    # Step 3: Generate answer using local LLM
    try:
        output = pipe(
            formatted_prompt,
            max_length=512,
            do_sample=False,
            repetition_penalty=1.1  # Prevents repetitive output
        )
        return output[0]['generated_text']
    except Exception as e:
        return f"[Error running local model] {e}"

# ============================================================================
# CLI TESTING INTERFACE
# ============================================================================

if __name__ == "__main__":
    print("RAG System - Interactive Mode")
    print("=" * 60)
    question = input("Ask a question: ")
    print("\nGenerating answer...\n")
    answer = generate_answer(question)
    print(f"Answer: {answer}")
