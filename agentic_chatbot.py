from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    RemoveMessage,
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from typing_extensions import TypedDict
from typing import Annotated

from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
DB_URI = os.getenv("DB_URI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    api_key=GOOGLE_API_KEY,
    
)

summary_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str


# -------------------------
# Chat Node
# -------------------------
def chat_node(state: ChatState):

    summary = state.get("summary", "")

    messages = []

    if summary:
        messages.append(
            {
                "role": "system",
                "content": f"Conversation summary:\n{summary}",
            }
        )

    messages.extend(state["messages"])

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }


# -------------------------
# Summarization Node
# -------------------------
def summarization_node(state: ChatState):

    existing_summary = state.get("summary", "")

    if existing_summary:
        prompt = f"""
Update this summary with the latest conversation.

Current Summary:
{existing_summary}

Keep it concise.
"""
    else:
        prompt = """
Summarize the conversation briefly.
Keep important facts and context.
"""

    messages_for_summary = state["messages"] + [
        HumanMessage(content=prompt)
    ]

    response = summary_llm.invoke(messages_for_summary)

    messages_to_delete = state["messages"][:-2]

    return {
        "summary": response.content,
        "messages": [
            RemoveMessage(id=m.id)
            for m in messages_to_delete
        ],
    }


# -------------------------
# Conditional Logic
# -------------------------
def should_summarize(state: ChatState):
    return len(state["messages"]) > 6


# -------------------------
# Graph
# -------------------------
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("summarize", summarization_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges(
    "chat_node",
    should_summarize,
    {
        True: "summarize",
        False: END,
    },
)

graph.add_edge("summarize", END)


# -------------------------
# Checkpointer
# -------------------------
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:

    checkpointer.setup()

    chatbot = graph.compile(
        checkpointer=checkpointer
    )

    thread_id = "1"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    while True:

        user_input = input("Ask Here: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        result = chatbot.invoke(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config=config,
        )
        message = result["messages"][-1]

        if isinstance(message.content, list):
            response = "".join(
                item["text"]
                for item in message.content
                if isinstance(item, dict) and item.get("type") == "text"
            )
            response = result["messages"][-1].text()
            print("AI:", response)

        else:
            print("AI:", message.content)