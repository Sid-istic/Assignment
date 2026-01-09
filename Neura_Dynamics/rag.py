import os
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_core.prompts import PromptTemplate

# Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "db_chroma")
COLLECTION_NAME = "policies"
MODEL_NAME = "google/flan-t5-base"

# Setup Embedding Function
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Prompt
PROMPT_TEMPLATE = """You are an accurate and strict company policy assistant. Your goal is to answer user questions truthfully using ONLY the provided context.

Instructions:
1. strict_grounding: Answer purely based on the 'Context' provided below. Do not use outside knowledge.
2. missing_info: If the answer is not explicitly stated in the context, respond with: "I cannot answer this based on the provided policies."
3. structure: Format your answer clearly. Use bullet points for lists.
4. tone: Professional and direct.

Context:
{context}

Question: {question}
Answer:"""

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE
)

# 1. Database Functions
# ==========================================
def get_collection():
    """Get the ChromaDB collection. Uses a fresh client to avoid stale handles."""
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)
    except Exception as e:
        print(f"DB Error: {e}")
        return None

def get_doc_count():
    """Helper to check if DB is populated."""
    col = get_collection()
    if col:
        return col.count()
    return 0

def retrieve_context(query, k=3):
    """Retrieve relevant chunks."""
    col = get_collection()
    if not col or col.count() == 0:
        return ""
    
    results = col.query(query_texts=[query], n_results=k)
    
    if not results['documents'] or not results['documents'][0]:
        return ""
    
    # Combine chunks
    chunks = results['documents'][0]
    return "\n---\n".join(chunks)

# 2. Model Loading (Cached)
# ==========================================
try:
    import streamlit as st
    cache_decorator = st.cache_resource
except ImportError:
    # Fallback for CLI
    def cache_decorator(func):
        return func

@cache_decorator
def load_llm_pipeline():
    print(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return pipeline(
        "text2text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_length=512
    )

pipeline_instance = load_llm_pipeline()

# 3. Generation Function
# ==========================================
def generate_answer(query):
    # 1. Retrieve
    context = retrieve_context(query)
    
    # Debug info for logs
    print(f"DEBUG: Query: {query}")
    print(f"DEBUG: Retrieved {len(context)} chars of context.")

    # 2. Format Prompt
    prompt = prompt_template.format(context=context, question=query)

    # 3. Generate
    # Using strict parameters to prevent looping/hallucination
    result = pipeline_instance(
        prompt, 
        max_length=512,
        do_sample=False, 
        repetition_penalty=2.0,       # Strong penalty for repeats
        no_repeat_ngram_size=3        # Hard block on 3-word loops
    )
    
    return result[0]['generated_text']

if __name__ == "__main__":
    # Simple CLI test
    q = input("Ask a question: ")
    print(generate_answer(q))
