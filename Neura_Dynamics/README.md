# RAG Policy Assistant

This is a Retrieval-Augmented Generation (RAG) system designed to answer questions about company policies (Refund, Cancellation, Shipping) accurately and without hallucination.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This will install `torch`, `transformers`, `langchain`, and `streamlit`.*

2. **Environment Variables**
   - Copy `.env.example` to `.env`

3. **Ingest Data**
   - This parses the policies in `data/` and builds the Chroma- **Run Interactive Test**:
   ```bash
   python rag.py
   ```

## Usage

- **Run Web Interface (Streamlit)**:
   This launches a chat UI in your browser.
   ```bash
   streamlit run app.py
   ```

- **Run Evaluation Script**:
   Tests the system against 5 predefined questions.
   ```bash
   python evaluate.py
   ```
- **Evaluation test being used** :
   ```bash
   "What is the return policy window?",
    "Do you offer free shipping?",
    "Can I cancel my order after it has shipped?",
    "Do you sell laptops?",  # Should be "I don't know"
    "I haven't received my refund after 10 days, what do I do?"
   ```
## Prompts
- **Prompt_Version_1** :
  Vague with no proper instructions.
  
  ```bash
      You are a helpful customer support assistant.
      Answer the user's question based on the provided policy documents.
      
      Context:
      {context}
      
      Question: {question}
      Answer:
   ```
- **Prompt_Version_2** :
  Detailed with how to answer.
  
   ```bash
      You are an accurate and strict company policy assistant. Your goal is to answer user questions truthfully using ONLY the provided context.
      
      Instructions:
      1. strict_grounding: Answer purely based on the 'Context' provided below. Do not use outside knowledge.
      2. missing_info: If the answer is not explicitly stated in the context, respond with: "I cannot answer this based on the provided policies."
      3. structure: Format your answer clearly. Use bullet points for lists.
      4. tone: Professional and direct.
      
      Context:
      {context}
      
      Question: {question}
      Answer:
   ```
## Architecture & Design

- **Chunking**: Uses `RecursiveSplitter` paragraph splitting (approx 300 chars) to preserve semantic clause meaning.
- **Vector Store**: `ChromaDB` with `all-MiniLM-L6-v2` embeddings for efficient local retrieval.
- **LLM**:
    - **Primary**: `google/flan-t5-base` (runs locally via Hugging Face Transformers).
    - **Prompting**: Managed via **LangChain** `PromptTemplate` to enforce strict grounding.
- **Interface**: Built with **Streamlit** for an easy-to-use chat experience.
