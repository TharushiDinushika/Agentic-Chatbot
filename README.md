# 🤖 Agentic Chatbot

A production-style conversational AI agent built with **LangGraph**, **LangChain**, and **Streamlit**. It can reason, call external tools, answer questions about uploaded PDFs (RAG), pause for human approval before sensitive actions, and remember every conversation across restarts. It is containerized with Docker and deployed to **AWS EC2** through a **GitHub Actions** CI/CD pipeline.

**🌐 Live demo:** [http://ec2-16-192-159-29.eu-north-1.compute.amazonaws.com:8501](http://ec2-16-192-159-29.eu-north-1.compute.amazonaws.com:8501)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Agentic workflow** | A LangGraph state graph where the LLM decides whether to answer directly or call a tool, then loops back to produce a final answer. |
| **Streaming responses** | Tokens stream into the chat bubble in real time; tool usage appears in a collapsible status box above the answer. |
| **Persistent memory** | Conversations are checkpointed to SQLite (`SqliteSaver`), so history survives page reloads and server restarts. |
| **Multi-thread chat** | Sidebar with all past conversations (newest first), a "New Chat" button, and auto-generated 2–5 word titles. |
| **RAG over PDFs** | Upload a PDF in the chat input; it is chunked, embedded, and stored in FAISS for retrieval by the agent. |
| **Tool calling** | Web search, calculator, live stock prices, live weather, and PDF retrieval. |
| **Human-in-the-loop (HITL)** | Stock purchases pause the graph with `interrupt()` and require an explicit Approve / Reject click. |
| **LLM fallback** | `gemini-3.5-flash` is the primary model with automatic fallback to `gemini-3.5-flash-lite`. |
| **Friendly error handling** | LLM failures are classified (timeout, connection, rephrase, generic) into safe user-facing messages. |
| **Observability** | LangSmith tracing for every run, including HITL resumes. |
| **CI/CD** | Push to `main` → validate → build and push Docker image → deploy to EC2 with health checks. |

---

## 🧰 Tools

| Tool | Purpose | Provider |
|---|---|---|
| `search_tool` | Current events and internet search | [Tavily](https://tavily.com) |
| `calculator` | Math expressions (restricted `eval`) | Built-in |
| `get_stock_price` | Latest stock quote by ticker | [Alpha Vantage](https://www.alphavantage.co) |
| `get_current_weather` | Current weather for any city | [OpenWeather](https://openweathermap.org) |
| `rag_tool` | Retrieve relevant chunks from uploaded PDFs | FAISS + Gemini embeddings |
| `purchase_stock` | Simulated stock purchase, **requires human approval** | Built-in (HITL) |

---

## 🛠️ Tech Stack

- **Orchestration:** LangGraph, LangChain
- **LLM:** Google Gemini (`gemini-3.5-flash` with `gemini-3.5-flash-lite` fallback)
- **Embeddings:** `gemini-embedding-001`
- **Vector store:** FAISS (CPU), PDFs loaded with `PyPDFLoader`, split with `RecursiveCharacterTextSplitter` (chunk size 1000, overlap 200, top-k 4)
- **Persistence:** SQLite via LangGraph `SqliteSaver`
- **Frontend:** Streamlit
- **Monitoring:** LangSmith
- **DevOps:** Docker, Docker Hub, GitHub Actions, AWS EC2 (self-hosted runner)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- API keys for Gemini, Tavily, OpenWeather, Alpha Vantage, and LangSmith

### 1. Clone and install

```bash
git clone <https://github.com/TharushiDinushika/Agentic-Chatbot.git
cd Agentic-Chatbot

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root and copy the content of the `.env.example` file and replace the values with your API keys.


### 3. Run locally

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

### Run with Docker

```bash
docker build -t agentic-chatbot .

docker run -d --name stapp -p 8501:8501 --env-file .env \
  -v "$(pwd)/database:/app/database" \
  agentic-chatbot
```

> Mounting `./database` keeps chat history and the FAISS index when the container is recreated.

---

## 💬 Example Prompts

- *"What's the weather in Colombo right now?"* → `get_current_weather`
- *"What is the current price of AAPL?"* → `get_stock_price`
- *"What is sqrt(144) * 12.5?"* → `calculator`
- *"Latest news about LangGraph"* → `search_tool`
- *Upload a PDF, then: "Summarize the key findings"* → `rag_tool`
- *"Buy 10 shares of TSLA"* → `purchase_stock`, then approve or reject in the UI

---

## ☁️ CI/CD Pipeline

Defined in `.github/workflows/` and triggered on every push to `main` (ignoring `README.md` changes) or manually via `workflow_dispatch`. A concurrency group ensures only one deployment runs at a time.

```
push to main
   │
   ▼
1. Continuous Integration     lint + unit tests (GitHub-hosted runner)
   │
   ▼
2. Build & Push Image         validate settings → Buildx → Docker Hub
   │                          tags: :main and :sha-<commit>, with GHA layer cache
   ▼
3. Deploy to AWS EC2          self-hosted runner on the EC2 instance
                              • log in to Docker Hub
                              • validate all application secrets
                              • pull latest image, replace the `stapp` container
                              • verify the container is running
                              • poll Streamlit's /_stcore/health endpoint
                              • show logs and prune unused images
```

### Required GitHub secrets

| Secret | Used for |
|---|---|
| `DOCKER_USERNAME`, `DOCKER_PASSWORD`, `IMAGE_NAME` | Docker Hub login and image naming |
| `GEMINI_API_KEY` | LLM and embeddings |
| `TAVILY_API_KEY` | Web search |
| `OPENWEATHER_API_KEY` | Weather tool |
| `ALPHAVANTAGE_API_KEY` | Stock tool |
| `LANGSMITH_TRACING`, `LANGSMITH_ENDPOINT`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | Tracing |

### EC2 setup notes

1. Launch an EC2 instance and install Docker.
2. Register the instance as a GitHub **self-hosted runner**.
3. Open inbound port **8501** in the instance's security group.

---

## 🗺️ Future Improvements
 
- **User management:** Add sign-up and login so each user has private chats and documents.
- **Persistent storage:** Keep data across redeployments with external databases such as PostgreSQL.
- **Rate limiting:** Limit usage per user to control API costs.
- **Human-in-the-loop actions:** Connect approvals to a real workflow, instead of only printing a simulated result.

---
