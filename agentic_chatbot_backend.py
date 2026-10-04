from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
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

from tools import (
    search_tool,
    calculator,
    get_stock_price,
    get_current_weather,
    purchase_stock,
)
from rag import rag_tool

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
tools = [
    search_tool,
    calculator,
    get_stock_price,
    get_current_weather,
    rag_tool,
    purchase_stock,
]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """LLM node that can answer directly or call an appropriate tool."""

    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"
            "Tool usage instructions:\n"
            "- Use `rag_tool` for questions about the uploaded PDF or document. "
            "Always retrieve relevant document content before answering PDF-related questions.\n"
            "- Use `search_tool` for current events, recent information, or information "
            "that requires an internet search.\n"
            "- Use `calculator` for mathematical calculations. Do not calculate complex "
            "expressions manually when the calculator is available.\n"
            "- Use `get_stock_price` when the user asks for the current price of a stock.\n"
            "- Use `get_current_weather` when the user asks about current weather for a location.\n\n"
            "Answer general questions directly when no tool is required. "
            "Do not invent information from the uploaded document. "
            "If the user asks about a PDF but no document is available, ask them to upload a PDF. "
            "After receiving a tool result, provide a clear and helpful final answer."
        )
    )

    messages = [system_message, *state["messages"]]

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

# chat titles and ordering
titles_conn = sqlite3.connect("database/chatbot.db", check_same_thread=False)
titles_conn.execute(
    "CREATE TABLE IF NOT EXISTS thread_titles "
    "(thread_id TEXT PRIMARY KEY, title TEXT NOT NULL)"
)
titles_conn.commit()
