import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import sqlite3

load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError(
        "GEMINI_API_KEY was not found. Add it to the workspace .env file and rerun this cell."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.2,
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState) -> dict[str, list[BaseMessage]]:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


conn = sqlite3.connect(database="database/chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn=conn)

builder = StateGraph(ChatState)
builder.add_node("chat_node", chat_node)
builder.add_edge(START, "chat_node")
builder.add_edge("chat_node", END)

chatbot = builder.compile(checkpointer=checkpoint)


def get_all_threads():
    all_threads = set()
    threads = checkpoint.list(None)
    for thread in threads:
        all_threads.add(thread.config["configurable"]["thread_id"])

    return list(all_threads)
