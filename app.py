from agentic_chatbot_backend import chatbot, checkpoint, llm, titles_conn
from chats import (
    get_all_threads,
    save_thread_title,
    generate_thread_title,
    ensure_titles,
)
from helpers import (
    generate_thread_id,
    add_thread,
    reset_chat,
    load_conversation,
    convert_messages,
    show_tools,
    sync_pending_interrupt,
    run_graph,
    finish_turn,
    resume_hitl_execution,
)
from langchain_core.messages import HumanMessage

from rag import ingest_rag_document

import streamlit as st
import tempfile
import os

# ------------------------------------------------------------------ page
st.set_page_config(page_title="Agentic Chatbot", page_icon="🤖")
st.title("Agentic Chatbot")

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "carry_tools" not in st.session_state:
    st.session_state["carry_tools"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads(checkpoint)  # newest first

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = ensure_titles(
        chatbot, titles_conn, st.session_state["chat_threads"]
    )

add_thread(st.session_state["thread_id"])
sync_pending_interrupt(st.session_state["thread_id"])

# ---------------------------------------------------------------- sidebar
st.sidebar.title("My Conversations")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

for tid in st.session_state["chat_threads"]:
    label = st.session_state["thread_titles"].get(tid, "New chat")
    is_current = tid == st.session_state["thread_id"]

    if st.sidebar.button(
        label,
        key=f"thread_{tid}",
        type="primary" if is_current else "secondary",
        use_container_width=True,
    ):
        st.session_state["thread_id"] = tid
        st.session_state["carry_tools"] = []
        st.session_state["message_history"] = convert_messages(load_conversation(tid))
        st.rerun()

# ------------------------------------------------------- chat history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            show_tools(message.get("tools"))
        st.markdown(message["content"])

# ------------------------------------------------------- HITL controls
pending_hitl = st.session_state.get("pending_hitl")

current_thread_has_pending_hitl = (
    pending_hitl is not None
    and pending_hitl.get("thread_id") == st.session_state["thread_id"]
)

if current_thread_has_pending_hitl:
    decision = st.session_state.pop("hitl_decision", None)

    if decision:
        resume_hitl_execution(decision)
    else:
        st.warning("🧑 Human approval required\n\n" f"{pending_hitl['prompt']}")

        approve_column, reject_column = st.columns(2)

        with approve_column:
            if st.button(
                "✅ Approve Purchase",
                key=f"approve_{st.session_state['thread_id']}",
                type="primary",
                use_container_width=True,
            ):
                st.session_state["hitl_decision"] = "yes"
                st.rerun()

        with reject_column:
            if st.button(
                "❌ Reject Purchase",
                key=f"reject_{st.session_state['thread_id']}",
                use_container_width=True,
            ):
                st.session_state["hitl_decision"] = "no"
                st.rerun()

# ---------------------------------------------------------- chat input
submission = st.chat_input(
    "Type your message here...",
    accept_file=True,
    file_type=["pdf"],
    disabled=current_thread_has_pending_hitl,
)

user_input = None

if submission:
    user_input = submission.text
    uploaded_files = submission.files

    if uploaded_files:
        uploaded_pdf = uploaded_files[0]
        temporary_file_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as temporary_file:
                temporary_file.write(uploaded_pdf.getvalue())
                temporary_file_path = temporary_file.name

            with st.spinner(f"Processing {uploaded_pdf.name}..."):
                ingest_rag_document(temporary_file_path)

            st.toast(f"{uploaded_pdf.name} processed successfully.", icon="✅")

        except Exception as error:
            st.error(f"PDF processing failed: {error}")

        finally:
            if temporary_file_path and os.path.exists(temporary_file_path):
                os.remove(temporary_file_path)

if user_input:
    thread_id = st.session_state["thread_id"]
    is_first_message = not st.session_state["message_history"]

    st.session_state["message_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_trace",
    }

    with st.chat_message("assistant"):
        text, tools = run_graph(
            {"messages": [HumanMessage(content=user_input)]}, config
        )

    finish_turn(thread_id, text, tools)

    # name the chat from its first message
    if is_first_message and thread_id not in st.session_state["thread_titles"]:
        title = generate_thread_title(llm, user_input)
        st.session_state["thread_titles"][thread_id] = title
        save_thread_title(titles_conn, thread_id, title)

    st.rerun()
