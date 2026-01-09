import streamlit as st
import os
# Import our robust helper functions
from rag import generate_answer, get_doc_count
# Import ingest function for auto-rebuild
from ingest import ingest

st.set_page_config(page_title="Policy Bot", page_icon="🤖")

st.title("🤖 Policy Q&A Assistant")

# 1. Sidebar Control
with st.sidebar:
    st.header("Admin System")
    count = get_doc_count()
    st.write(f"**Knowledge Base Status:** {count} documents chunks")
    
    if st.button("Rebuild Knowledge Base"):
        with st.spinner("Rebuilding..."):
            ingest()
            st.success("Analysis Complete!")
            st.cache_resource.clear()
            st.rerun()

# 2. Auto-Initialization Logic
# If the DB is empty (first run on Cloud), build it.
if get_doc_count() == 0:
    with st.spinner("Initializing System (First Run)..."):
        ingest()
        st.success("System Ready!")
        st.rerun()

# 3. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
if prompt := st.chat_input("Ask about refunds, shipping, or cancellations..."):
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_answer(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# 4. Debug Section (Optional, good for troubleshooting Cloud)
with st.expander("Debug Info"):
    st.text(f"DB Path: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db_chroma')}")
    st.text(f"Doc Count: {get_doc_count()}")
