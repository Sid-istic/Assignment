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
   - This parses the policies in `data/` and builds the Chroma Vector DB.
   ```bash
   python -m src.ingest
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

## Architecture & Design

- **Chunking**: Uses `RecursiveLike` paragraph splitting (approx 500 chars) to preserve semantic clause meaning.
- **Vector Store**: `ChromaDB` with `all-MiniLM-L6-v2` embeddings for efficient local retrieval.
- **LLM**:
    - **Primary**: `google/flan-t5-base` (runs locally via Hugging Face Transformers).
    - **Prompting**: Managed via **LangChain** `PromptTemplate` to enforce strict grounding.
- **Interface**: Built with **Streamlit** for an easy-to-use chat experience.
