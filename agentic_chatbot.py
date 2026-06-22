from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import Annotated, TypedDict
from dotenv import load_dotenv
import os 

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    api_key= GOOGLE_API_KEY
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    query = state['messages']
    response = llm.invoke(query)
    return{
        "messages":[response]
    }

graph = StateGraph(ChatState)
graph.add_node("Chat_Node", chat_node)

graph.add_edge(START, "Chat_Node")
graph.add_edge("Chat_Node", END)

checkpoint = InMemorySaver()

thread_id = "1"
config= {"configurable":{"thread_id":thread_id}}

chatbot = graph.compile(checkpointer=checkpoint)