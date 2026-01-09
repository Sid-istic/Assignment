# RAG Policy Assistant

This is a Retrieval-Augmented Generation (RAG) system designed to answer questions about company policies (Refund, Cancellation, Shipping) accurately and without hallucination.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This will install `torch`, `transformers`, and `langchain`, which are required for the local model.*

2. **Environment Variables**
   - Copy `.env.example` to `.env`

3. **Ingest Data**
   - This parses the policies in `data/` and builds the Chroma Vector DB.
   ```bash
   python -m src.ingest
   ```

## Usage

- **Run Evaluation Script**:
   Tests the system against 5 predefined questions.
   ```bash
   python evaluate.py
   ```
   *Note: The first run will download the model (~900MB).*

- **Run Interactive Test**:
   ```bash
   python -m src.rag
   ```

## Architecture & Design

- **Chunking**: Uses `RecursiveLike` paragraph splitting (approx 500 chars) to preserve semantic clause meaning.
- **Vector Store**: `ChromaDB` with `all-MiniLM-L6-v2` embeddings for efficient local retrieval.
- **LLM**:
    - **Primary**: `google/flan-t5-base` (runs locally via Hugging Face Transformers).
    - **Prompting**: Managed via **LangChain** `PromptTemplate` to enforce strict grounding (answering ONLY from context).
- **RAG Pipeline**:
    1. Retrieve top-3 chunks based on query similarity.
    2. Format prompt using LangChain template.
    3. Generate answer using local Seq2Seq model.

## Evaluation Results

Run `evaluate.py` to observe:
1. **Accuracy**: Correctly answers policy questions (e.g., "30 days" for refunds).
2. **Grounding**: strictly refuses to answer out-of-scope questions (e.g., "Do you sell laptops?").
3. **Edge Cases**: Handles scenarios where info is spread across documents or missing.
