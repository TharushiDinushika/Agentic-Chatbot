from agentic_chatbot_backend import chatbot, get_all_threads
from langchain_core.messages import AIMessage, HumanMessage

import streamlit as st
import uuid


def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):

    # prevent the same thread from being added multiple times
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    # generate and assign a new threas ID
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


st.title("Agentic Chatbot with LangGraph")


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

# add the current thread to the conversation list
add_thread(st.session_state["thread_id"])


st.sidebar.title("My Conversations")

if st.sidebar.button("New Chat"):

    # Reset the current chat and create a new thread
    reset_chat()

    # Rerun the Streamlit app to update the interface
    st.rerun()

# display all conversation threads in reverse order
# this shows the newest conversation first
for thread_id in st.session_state["chat_threads"][::-1]:

    # create one sidebar button for every conversation
    if st.sidebar.button(str(thread_id), key=thread_id):

        # set the selected thread as the current thread
        st.session_state["thread_id"] = thread_id

        # load the messages saved under the selected thread
        messages = load_conversation(thread_id)

        # temporary list for converting LangChain messages
        # into Streamlit's required message format
        temp_messages = []

        # loop through all saved messages
        for message in messages:

            # check whether the message was sent by the user
            if isinstance(message, HumanMessage):
                role = "user"

            # check whether the message was sent by the AI
            elif isinstance(message, AIMessage):
                role = "assistant"

            # ignore other message types, such as ToolMessage
            else:
                continue

            # convert the LangChain message into a dictionary
            temp_messages.append({"role": role, "content": message.text})

        # replace the current UI history with the selected conversation
        st.session_state["message_history"] = temp_messages

        # rerun the application to display the loaded messages
        st.rerun()


# loading the conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])


user_input = st.chat_input("Type your message here...")

if user_input:
    # first add the message to the session state
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # pass the current thread ID to LangGraph
    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    with st.chat_message("assistant"):

        ai_message = st.write_stream(
            message_chunk.text
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            )
            # display only AI message
            if isinstance(message_chunk, AIMessage)
        )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
