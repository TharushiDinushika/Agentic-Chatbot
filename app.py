from agentic_chatbot_backend import chatbot, get_all_threads
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from rag import ingest_rag_document

import streamlit as st
import uuid
import tempfile
import os


def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):

    # prevent the same thread from being added multiple times
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    # generate and assign a new thread ID
    st.session_state["thread_id"] = generate_thread_id()

    # clear the current chat messages from ui
    st.session_state["message_history"] = []

    # add the new thread to the conversation list
    add_thread(st.session_state["thread_id"])


# load a previous conversation from the LangGraph checkpointer
def load_conversation(thread_id):

    # get the saved state for the selected thread
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})

    # return saved messages
    # return an empty list if no messages are available
    return state.values.get("messages", [])


def convert_messages(messages):
    """Convert saved LangChain messages into Streamlit UI messages,
    attaching any tool calls to the assistant answer that follows them."""
    ui_messages = []
    pending_tools = []  # tools used before the next assistant answer

    for message in messages:
        if isinstance(message, HumanMessage):
            ui_messages.append({"role": "user", "content": message.text})

        elif isinstance(message, ToolMessage):
            pending_tools.append({"name": message.name or "tool"})

        elif isinstance(message, AIMessage):
            # AI messages that only request tool calls have no text, skip them
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


def get_pending_interrupt(thread_id):
    """
    Return the first unresolved LangGraph interrupt for a thread.

    Returns:
        The pending Interrupt object, or None.
    """

    config = {"configurable": {"thread_id": thread_id}}

    try:

        # read the current checkpoint state
        state_snapshot = chatbot.get_state(config)

        direct_interrupts = getattr(state_snapshot, "interrupts", ()) or ()

        if direct_interrupts:
            return direct_interrupts[0]

        # other LangGraph versions store interrupts inside tasks
        tasks = getattr(state_snapshot, "tasks", ()) or ()

        for task in tasks:

            task_interrupts = getattr(task, "interrupts", ()) or ()

            if task_interrupts:
                return task_interrupts[0]

    except Exception:

        # newly created thread may not have a checkpoint yet
        return None

    return None


def save_pending_interrupt(thread_id, interrupt_object):
    """
    Save the pending interrupt information inside Streamlit state.
    """

    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value),
    }


def sync_pending_interrupt(thread_id):
    """
    Synchronize Streamlit HITL state with the LangGraph checkpoint.

    This allows a pending approval request to reappear after:
    - a Streamlit rerun
    - a browser refresh
    - switching between conversations
    """

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


def resume_hitl_execution(decision):
    """
    Resume an interrupted LangGraph execution.

    Args:
        decision:
            "yes" approves the stock purchase.
            "no" rejects the stock purchase.
    """

    pending_hitl = st.session_state.get("pending_hitl")

    if not pending_hitl:

        st.warning("There is no pending action to approve or reject.")

        return

    # Get the thread that originally triggered the interrupt
    interrupted_thread_id = pending_hitl["thread_id"]

    # The same thread ID must be used when resuming
    resume_config = {
        "configurable": {"thread_id": interrupted_thread_id},
        "metadata": {"thread_id": interrupted_thread_id},
        "run_name": "hitl_resume_trace",
    }

    try:

        # display the resumed response
        with st.chat_message("assistant"):

            status_holder = {
                "box": st.status("🔄 Resuming the requested action...", expanded=True)
            }

            def resumed_ai_only_stream():

                # resume the graph with the human decision
                for message_chunk, metadata in chatbot.stream(
                    Command(resume=decision),
                    config=resume_config,
                    stream_mode="messages",
                ):

                    # update tool execution status
                    if isinstance(message_chunk, ToolMessage):

                        tool_name = getattr(message_chunk, "name", "tool")

                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                    # stream only assistant-generated text
                    if isinstance(message_chunk, AIMessage):

                        if message_chunk.text:
                            yield message_chunk.text

            # display the streamed final answer
            resumed_ai_message = st.write_stream(resumed_ai_only_stream())

            # check whether another interrupt occurred
            next_interrupt = get_pending_interrupt(interrupted_thread_id)

            if next_interrupt is not None:

                save_pending_interrupt(interrupted_thread_id, next_interrupt)

                status_holder["box"].update(
                    label="⚠️ Another approval is required",
                    state="complete",
                    expanded=False,
                )

            else:

                # no more pending approval
                st.session_state["pending_hitl"] = None

                status_holder["box"].update(
                    label="✅ Action completed", state="complete", expanded=False
                )

        # save the assistant response in Streamlit UI history
        if resumed_ai_message:

            st.session_state["message_history"].append(
                {"role": "assistant", "content": resumed_ai_message}
            )

        # Rerun so the response appears in normal chat order
        st.rerun()

    except Exception as error:

        st.error(f"Could not resume the requested action: {error}")


st.set_page_config(page_title="Agentic Chatbot", page_icon="🤖")

st.title("Agentic Chatbot")


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

# add the current thread to the conversation list
add_thread(st.session_state["thread_id"])

# recover pending approval after page refresh or rerun
sync_pending_interrupt(st.session_state["thread_id"])

st.sidebar.title("My Conversations")

if st.sidebar.button("New Chat"):

    reset_chat()

    st.rerun()

# display all conversation threads in reverse order
for thread_id in st.session_state["chat_threads"][::-1]:

    # create one sidebar button for every conversation
    if st.sidebar.button(str(thread_id), key=thread_id):

        # set the selected thread as the current thread
        st.session_state["thread_id"] = thread_id

        # load the messages saved under the selected thread
        messages = load_conversation(thread_id)

        # replace the current UI history with the selected conversation
        st.session_state["message_history"] = convert_messages(messages)

        # rerun the application to display the loaded messages
        st.rerun()


# loading the conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            show_tools(message.get("tools"))
        st.markdown(message["content"])

# get the currently pending approval request
pending_hitl = st.session_state.get("pending_hitl")

# check whether the pending approval belongs to the currently selected conversation
current_thread_has_pending_hitl = (
    pending_hitl is not None
    and pending_hitl.get("thread_id") == st.session_state["thread_id"]
)


# display approval controls
if current_thread_has_pending_hitl:

    st.warning("🧑 Human approval required\n\n" f"{pending_hitl['prompt']}")

    approve_column, reject_column = st.columns(2)

    with approve_column:

        if st.button(
            "✅ Approve Purchase",
            key=f"approve_{st.session_state['thread_id']}",
            type="primary",
            use_container_width=True,
        ):

            resume_hitl_execution("yes")

    with reject_column:

        if st.button(
            "❌ Reject Purchase",
            key=f"reject_{st.session_state['thread_id']}",
            use_container_width=True,
        ):

            resume_hitl_execution("no")


submission = st.chat_input(
    "Type your message here...",
    accept_file=True,
    file_type=["pdf"],
    # disable input while waiting for human approval
    disabled=current_thread_has_pending_hitl,
)


# default user input value
user_input = None


# Process the submitted text and PDF
if submission:

    user_input = submission.text

    # get the uploaded files
    uploaded_files = submission.files

    # process the uploaded PDF if one was attached
    if uploaded_files:

        uploaded_pdf = uploaded_files[0]

        # store the temporary file path
        temporary_file_path = None

        try:

            # save the uploaded PDF as a temporary local file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as temporary_file:

                temporary_file.write(uploaded_pdf.getvalue())

                temporary_file_path = temporary_file.name

            # call the existing backend RAG ingestion function
            with st.spinner(f"Processing {uploaded_pdf.name}..."):

                ingest_rag_document(temporary_file_path)

            # display PDF processing confirmation
            st.toast(f"{uploaded_pdf.name} processed successfully.", icon="✅")

        except Exception as error:

            # display PDF processing error
            st.error(f"PDF processing failed: {error}")

        finally:

            # delete the temporary PDF after indexing
            if temporary_file_path and os.path.exists(temporary_file_path):
                os.remove(temporary_file_path)


if user_input:

    # save the user's message in Streamlit session state
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # pass the current thread ID to LangGraph
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_trace",
    }

    with st.chat_message("assistant"):
        # use a mutable holder so the generator can modify it
        status_holder = {"box": None}

        # collect tools used in this turn so they can be shown after reloads
        tools_used = []

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")

                    tools_used.append({"name": tool_name})

                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # stream only assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.text

        ai_message = st.write_stream(ai_only_stream())

        # finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message, "tools": tools_used}
    )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message, "tools": tools_used}
    )

    # check the checkpoint AFTER the stream, since the interrupt may have just happened
    new_interrupt = get_pending_interrupt(st.session_state["thread_id"])

    if new_interrupt is not None:
        save_pending_interrupt(st.session_state["thread_id"], new_interrupt)
        st.rerun()
