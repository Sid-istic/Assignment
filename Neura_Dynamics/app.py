import streamlit as st
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.rag import generate_answer

st.set_page_config(page_title="Policy RAG Bot", page_icon="🤖")

# Check if DB exists, if not, ingest data
import os
if not os.path.exists("db_chroma"):
    with st.spinner("Building Knowledge Base... (This happens only once)"):
        from src.ingest import ingest
        ingest()
        st.success("Knowledge Base Built!")

st.title("🤖 Policy Assistant")
st.markdown("Ask questions about **Refunds**, **Cancellations**, or **Shipping**.")


# Chat history state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("How can I return an item?"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_answer(prompt)
            st.markdown(response)
    
    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
