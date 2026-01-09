import streamlit as st
import os
from rag import generate_answer, get_doc_count

st.set_page_config(
    page_title="Policy RAG Bot",
    page_icon="🤖",
    layout="centered"
)

with st.sidebar:
    st.header("⚙️ Settings")
    
    # Display database status
    doc_count = get_doc_count()
    st.metric("Documents in Database", doc_count)
    
    # Manual rebuild button
    if st.button("🔄 Rebuild Knowledge Base"):
        with st.spinner("Rebuilding database..."):
            from ingest import ingest
            ingest()
            st.cache_resource.clear()  # Clear model cache
            st.success("✅ Database rebuilt successfully!")
            st.rerun()  # Refresh to show new count

# Check if database is empty and auto-initialize if needed
# Use session state to prevent re-running on every rerun
if "db_initialized" not in st.session_state:
    st.session_state.db_initialized = False

if not st.session_state.db_initialized and get_doc_count() == 0:
    with st.spinner("🔨 Building Knowledge Base... (First time setup)"):
        from ingest import ingest
        ingest()
        st.session_state.db_initialized = True
        st.success("✅ Knowledge Base initialized!")
        st.rerun()  # Refresh to show updated count

st.title("🤖 Policy Assistant")
st.markdown(
    "Ask questions about **Refunds**, **Cancellations**, or **Shipping** policies."
)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("How can I return an item?"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_answer(prompt)
            st.markdown(response)
    
    # Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
