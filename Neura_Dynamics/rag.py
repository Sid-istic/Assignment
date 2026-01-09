import os
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


load_dotenv()



# Determine the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "db_chroma")
COLLECTION_NAME = "policies"
TOP_K = 3  # Number of relevant chunks to retrieve
MODEL_NAME = "google/flan-t5-base"


PROMPT_TEMPLATE = """You are a helpful customer support assistant.
Answer the user's question based on the provided policy documents.

Context:
{context}

Question: {question}
Answer:"""

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

    try:
        collection = get_collection()
        return collection.count()
    except Exception as e:
        # If collection doesn't exist or is corrupted, return 0
        print(f"Warning: Could not get doc count: {e}")
        return 0


try:
    import streamlit as st
    cache_decorator = st.cache_resource
except ImportError:
    # Dummy decorator for CLI usage
    def cache_decorator(func):
        return func

@cache_decorator
def load_model():

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
    
    # DEBUG: Log what we retrieved
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"RETRIEVED CONTEXT LENGTH: {len(context)} chars")
    if context:
        print(f"CONTEXT PREVIEW: {context[:200]}...")
    else:
        print("⚠️ WARNING: NO CONTEXT RETRIEVED!")
    print(f"{'='*60}\n")
    
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

