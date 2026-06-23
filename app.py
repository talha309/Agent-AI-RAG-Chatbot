import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessageChunk
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def get_chatbot_app():
    # Aapke corrected chatbot graph ko direct import karein
    from agentic_chatbot import chatbot
    return chatbot

try:
    chatbot_app = get_chatbot_app()
except Exception as db_err:
    st.error(f"Database/Graph Loading Error: {str(db_err)}")
    st.stop()

st.set_page_config(page_title="Agentic Chatbot", page_icon="🤖", layout="centered")
st.title("Agentic Chatbot with LangGraph & PostgreSQL")
st.subheader("Persistent Memory Enabled via Docker Postgres")

CONFIG = {
    "configurable": {
        "thread_id": "streamlit-session-thread-1"
    }
}

if "message_history" not in st.session_state:
    st.session_state.message_history = []

for msg in st.session_state.message_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask Here...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.message_history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            # Graph streaming call
            for message_chunk, metadata in chatbot_app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, AIMessageChunk):
                    content = message_chunk.content

                    if isinstance(content, str):
                        full_response += content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                full_response += block.get("text", "")
                            elif hasattr(block, "text"):
                                full_response += block.text

                    placeholder.markdown(full_response + "▌")
            
            placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"Streaming error occurred: {str(e)}")
            full_response = "Sorry, I encountered an error processing your request."
            placeholder.markdown(full_response)

    st.session_state.message_history.append({"role": "assistant", "content": full_response})
    st.rerun()