from langchain_core.messages import HumanMessage, SystemMessage


def get_all_threads(checkpoint):
    """Thread IDs sorted by most recent activity."""
    latest = {}
    for item in checkpoint.list(None):
        thread_id = item.config["configurable"]["thread_id"]
        ts = item.checkpoint["ts"]  # ISO timestamp of the checkpoint
        if thread_id not in latest or ts > latest[thread_id]:
            latest[thread_id] = ts

    return [t for t, _ in sorted(latest.items(), key=lambda x: x[1], reverse=True)]


def get_thread_titles(titles_conn) -> dict:
    rows = titles_conn.execute("SELECT thread_id, title FROM thread_titles").fetchall()
    return {thread_id: title for thread_id, title in rows}


def save_thread_title(titles_conn, thread_id: str, title: str):
    titles_conn.execute(
        "INSERT OR REPLACE INTO thread_titles (thread_id, title) VALUES (?, ?)",
        (thread_id, title),
    )
    titles_conn.commit()


def generate_thread_title(llm, first_message: str) -> str:
    """Ask the LLM for a short title; fall back to the truncated message."""
    try:
        result = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Write a short title (2-5 words) for a chat that starts "
                        "with the user message below. Reply with only the title, "
                        "no quotes and no punctuation at the end."
                    )
                ),
                HumanMessage(content=first_message[:500]),
            ]
        )
        title = result.text.strip().strip("\"'").strip()
    except Exception:
        title = ""

    return (title or first_message.strip())[:50] or "New chat"


def ensure_titles(chatbot, titles_conn, thread_ids) -> dict:
    """Give old threads (created before titles existed) a title from their first message."""
    titles = get_thread_titles(titles_conn)
    for thread_id in thread_ids:
        if thread_id in titles:
            continue
        state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        first = next((m.text for m in messages if isinstance(m, HumanMessage)), None)
        if first:
            titles[thread_id] = first.strip()[:40]
            save_thread_title(titles_conn, thread_id, titles[thread_id])
    return titles
