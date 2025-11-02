#!/usr/bin/env python3
"""
Streamlit UI for Enterprise Knowledge Assistant
MVP interface for querying the RAG system.
"""

import streamlit as st
import requests
import os
from datetime import datetime
from typing import List, Dict, Any
from src.config import config  # Import config to get API_URL

# Use the API_URL from the configuration
API_URL = config.API_URL

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Enterprise Knowledge Assistant")
st.markdown("Ask questions about company documentation. Answers are sourced exclusively from internal documents.")

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_data(ttl=10)
def check_api_health() -> bool:
    """Check if the API service is available."""
    try:
        # Use the API_URL (e.g., http://api:8000/health)
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200 and response.json().get("status") == "healthy"
    except:
        return False


api_healthy = check_api_health()

if not api_healthy:
    st.error(f"⚠️ API service is not available at `{API_URL}`. Please ensure the `api` service is running.")
    st.info("Run `docker-compose logs -f api` to check the status.")
    st.stop()

with st.sidebar:
    st.header("Conversation Controls")

    if st.button("🔄 New Conversation"):
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.header("About")
    st.markdown("""
    This assistant answers questions using **only** internal company documentation.

    **Features:**
    - Zero hallucination guarantee
    - Source attribution
    - Multi-turn conversation support
    - Runs 100% locally
    """)

    if st.session_state.conversation_id:
        st.divider()
        st.caption(f"Conversation ID: `{st.session_state.conversation_id[:8]}...`")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                st.markdown("**📎 Sources:**")
                for source in message["sources"]:
                    if source['version']:
                        st.markdown(f"- [{source['title']}]({source['url']}) (v{source['version']})")
                    else:
                        st.markdown(f"- [{source['title']}]({source['url']})")
            elif "Error" not in message["content"]:
                st.caption("ℹ️ No relevant documentation found for this question.")

# Handle new user input
if prompt := st.chat_input("Ask a question about company documentation..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 Searching documentation...")

        try:
            payload = {
                "query": prompt,
                "conversation_id": st.session_state.conversation_id
            }

            response = requests.post(
                f"{API_URL}/ask",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()

                answer = result["answer"]
                sources = result["sources"]
                conversation_id = result["conversation_id"]

                if st.session_state.conversation_id is None:
                    st.session_state.conversation_id = conversation_id

                message_placeholder.markdown(answer)

                if sources:
                    st.markdown("**📎 Sources:**")
                    for source in sources:
                        if source['version']:
                            st.markdown(f"- [{source['title']}]({source['url']}) (v{source['version']})")
                        else:
                            st.markdown(f"- [{source['title']}]({source['url']})")
                else:
                    st.caption("ℹ️ No relevant documentation found for this question.")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            else:
                error_message = f"⚠️ Error: {response.status_code} - {response.text}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": []
                })

        except requests.exceptions.Timeout:
            error_message = "⚠️ Request timed out. The question may be too complex or the service is overloaded."
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": []
            })
        except Exception as e:
            error_message = f"⚠️ An error occurred: {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": []
            })