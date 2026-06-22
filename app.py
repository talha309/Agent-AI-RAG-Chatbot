from agentic_chatbot import chatbot
from langchain_core.messages import HumanMessage
import streamlit as st

CONFIG = {
    "configurable": {
        "thread_id": "thread-1"
    }
}

st.title("Agentic Chatbot with LangGraph")

if "message_history" not in st.session_state:
    st.session_state.message_history = []

# Display history
for msg in st.session_state.message_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask Here")

if user_input:

    # User message
    st.session_state.message_history.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant message
    with st.chat_message("assistant"):

        placeholder = st.empty()
        full_response = ""

        for message_chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages",
        ):

            content = message_chunk.content

            # String response
            if isinstance(content, str):
                full_response += content

            # Gemini content blocks
            elif isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                    ):
                        full_response += block.get("text", "")

            placeholder.markdown(full_response)

    st.session_state.message_history.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )