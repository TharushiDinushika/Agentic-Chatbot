import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from agentic_chatbot_backend import chatbot


# ------------------------------------------------------------ threads
def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    """Insert a new thread at the top (list is kept newest-first)."""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].insert(0, thread_id)


def touch_thread(thread_id):
    """Move a thread to the top after it receives activity."""
    threads = st.session_state["chat_threads"]
    if thread_id in threads:
        threads.remove(thread_id)
    threads.insert(0, thread_id)


def reset_chat():
    # don't pile up empty chats
    if not st.session_state["message_history"]:
        return
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    st.session_state["carry_tools"] = []
    add_thread(st.session_state["thread_id"])


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ----------------------------------------------------------- messages
def convert_messages(messages):
    """Convert saved LangChain messages into Streamlit UI messages,
    attaching any tool calls to the assistant answer that follows them."""
    ui_messages = []
    pending_tools = []

    for message in messages:
        if isinstance(message, HumanMessage):
            ui_messages.append({"role": "user", "content": message.text})

        elif isinstance(message, ToolMessage):
            pending_tools.append({"name": message.name or "tool"})

        elif isinstance(message, AIMessage):
            if not message.text:
                continue
            ui_messages.append(
                {"role": "assistant", "content": message.text, "tools": pending_tools}
            )
            pending_tools = []

    return ui_messages


def show_tools(tools):
    """Render the tools used for an assistant message in a collapsed box."""
    if tools:
        with st.status(f"Used {len(tools)} tool(s)", state="complete", expanded=False):
            for t in tools:
                st.markdown(f"- `{t['name']}`")


# ---------------------------------------------------------- interrupts
def get_pending_interrupt(thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state_snapshot = chatbot.get_state(config)

        direct_interrupts = getattr(state_snapshot, "interrupts", ()) or ()
        if direct_interrupts:
            return direct_interrupts[0]

        for task in getattr(state_snapshot, "tasks", ()) or ():
            task_interrupts = getattr(task, "interrupts", ()) or ()
            if task_interrupts:
                return task_interrupts[0]
    except Exception:
        return None
    return None


def save_pending_interrupt(thread_id, interrupt_object):
    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value),
    }


def sync_pending_interrupt(thread_id):
    pending_interrupt = get_pending_interrupt(thread_id)

    if pending_interrupt is not None:
        save_pending_interrupt(thread_id, pending_interrupt)
    else:
        current_pending = st.session_state.get("pending_hitl")
        if (
            current_pending is not None
            and current_pending.get("thread_id") == thread_id
        ):
            st.session_state["pending_hitl"] = None


# ----------------------------------------------------- graph streaming
def run_graph(graph_input, config):
    """
    Stream one graph run inside the current assistant bubble.
    Used for both normal messages and HITL resume.

    Returns (final_text, tools_used).
    """
    tools_used = []

    # created BEFORE streaming so the tool box sits above the answer text
    tool_area = st.container()
    box = {"status": None}

    def register_tool(name):
        if name in [t["name"] for t in tools_used]:
            return
        tools_used.append({"name": name})

        with tool_area:
            if box["status"] is None:
                box["status"] = st.status(f"Using `{name}` …", expanded=False)
            else:
                box["status"].update(
                    label=f"Using `{name}` …", state="running", expanded=False
                )

    def stream_text():
        for chunk, _metadata in chatbot.stream(
            graph_input, config=config, stream_mode="messages"
        ):
            if isinstance(chunk, AIMessage):
                for call_chunk in getattr(chunk, "tool_call_chunks", None) or []:
                    if call_chunk.get("name"):
                        register_tool(call_chunk["name"])

                if chunk.text:
                    yield chunk.text

            elif isinstance(chunk, ToolMessage):
                register_tool(getattr(chunk, "name", None) or "tool")

    text = st.write_stream(stream_text())

    if box["status"] is not None:
        box["status"].update(
            label=f"Used {len(tools_used)} tool(s)", state="complete", expanded=False
        )

    return text, tools_used


def finish_turn(thread_id, text, tools):
    """Update history, pending interrupt state and sidebar ordering after a run."""
    all_tools = st.session_state.get("carry_tools", []) + tools
    st.session_state["carry_tools"] = []

    interrupt_object = get_pending_interrupt(thread_id)
    if interrupt_object is not None:
        save_pending_interrupt(thread_id, interrupt_object)
    else:
        st.session_state["pending_hitl"] = None

    if text:
        st.session_state["message_history"].append(
            {"role": "assistant", "content": text, "tools": all_tools}
        )
    else:
        # tool-call-only turn (waiting for approval): no empty bubble,
        # attach these tools to the next answer instead
        st.session_state["carry_tools"] = all_tools

    touch_thread(thread_id)


def resume_hitl_execution(decision):
    pending_hitl = st.session_state.get("pending_hitl")
    if not pending_hitl:
        return

    thread_id = pending_hitl["thread_id"]
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "hitl_resume_trace",
    }

    try:
        with st.chat_message("assistant"):
            text, tools = run_graph(Command(resume=decision), config)
        finish_turn(thread_id, text, tools)
    except Exception as error:
        st.error(f"Could not resume the requested action: {error}")
        return

    st.rerun()
