from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from typing import Any
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import requests
import math
import os

import sqlite3

from tools import search_tool, calculator, get_stock_price, get_current_weather

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError(
        "GEMINI_API_KEY was not found. Add it to the workspace .env file and rerun this cell."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.2,
)

# bind tools to the LLM
tools = [search_tool, calculator, get_stock_price, get_current_weather]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)


conn = sqlite3.connect(database="database/chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpoint)


def get_all_threads():
    all_threads = set()
    threads = checkpoint.list(None)
    for thread in threads:
        all_threads.add(thread.config["configurable"]["thread_id"])

    return list(all_threads)
