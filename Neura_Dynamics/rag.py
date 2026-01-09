import os
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Config
DB_DIR = "db_chroma"
COLLECTION_NAME = "policies"
TOP_K = 3
MODEL_NAME = "google/flan-t5-base"


template = """Use the following pieces of context to answer the question at the end. Answer in detail and complete sentences. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""



prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# Setup DB Client
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

# Simple caching wrapper
# If running in Streamlit, use st.cache_resource.
# If running in CLI, just run the function.
try:
    import streamlit as st
    cache_decorator = st.cache_resource
except ImportError:
    # Dummy decorator that does nothing
    def cache_decorator(func):
        return func

@cache_decorator
def load_model():
    print(f"Loading local model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)

pipe = load_model()

def retrieve_context(query):
    """Retrieves top-k relevant chunks."""
    results = collection.query(
        query_texts=[query],
        n_results=TOP_K
    )
    
    docs = results['documents'][0]
    return "\n---\n".join(docs)

def generate_answer(query):
    """
    Main RAG function using local Hugging Face model + LangChain Prompt.
    """
    context = retrieve_context(query)
    
    # Generate prompt using LangChain
    formatted_prompt = prompt_template.format(context=context, question=query)
    
    try:
        # Run generation with specific parameters
        output = pipe(formatted_prompt, max_length=512, do_sample=False, repetition_penalty=1.1)
        return output[0]['generated_text']
    except Exception as e:
        return f"[Error running local model] {e}"

if __name__ == "__main__":
    q = input("Ask a question: ")
    print(generate_answer(q))
