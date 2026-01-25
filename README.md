# 🚗 AutoFinance AI Agent ("Financial Co-Pilot")

> **A State-of-the-Art Agentic Workflow for the Egyptian Vehicle Market**

The **AutoFinance AI Agent** is an autonomous "Financial Co-Pilot" designed to bridge the gap between vehicle discovery and financing in Egypt. It guides users through a complete journey: finding cars on fragmented marketplaces (Hatla2ee, Dubizzle), retrieving specific credit policies via RAG, calculating precise monthly installments, and submitting high-intent applications for back-office review.

---

## 🚀 Key Features

*   **🔍 Market Discovery**: Real-time aggregation of vehicle listings from Egyptian marketplaces using Tavily API.
*   **🧠 Intelligent Routing**: Context-aware routing between onboarding, searching, profiling, and specific inquiries.
*   **🛡️ Policy Enforcement (RAG)**: Retrieves and applies internal credit policies (interest rates, DBR limits) based on user profile and vehicle age.
*   **🧮 Loan Quotation**: Precise calculation of monthly installments using the PMT formula and affordability checks.
*   **💾 State Persistence**: Remembers user context (selected car, income, employment) across sessions using SQLite checkpoints.
*   **🚦 Human-in-the-Loop**: Explicit validation steps before executing searches or submitting sensitive applications.

---

## 🏗️ Technical Architecture

This project is built on the **DataRobot Agentic Workflow** template using **LangGraph**.

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Orchestration** | **LangGraph** | State-based graph managing the 5-phase user journey. |
| **Reasoning** | **DataRobot LLM Gateway** | Access to hosted LLMs (e.g., GPT-4o) with reliability guards. |
| **Memory** | **SQLite** | `langgraph.checkpoint.sqlite` for session state persistence. |
| **Search** | **Tavily API** | Optimized search for parsing unstructured vehicle listing data. |
| **Compute** | **DataRobot Custom Models** | Serverless runtime for hosting the agent logic. |

---

## 📂 Project Structure

> ⚠️ **Note**: The structure below is the **target architecture** and will be built **step by step**. Each component will be developed and tested individually before moving to the next.

```
agent_langgraph/
├── agentic_workflow/
│   ├── agent.py              # 🧠 Main StateGraph definition & Agent class
│   ├── custom.py             # 🔌 DataRobot hooks & Persistence initialization
│   ├── models.py             # 📦 Pydantic data models (State, Vehicle, Quote)
│   ├── config.py             # ⚙️ Configuration & Environment variables
│   ├── nodes/                # 📍 Graph Nodes (Functional Units)
│   │   ├── router.py         #    → Intent routing
│   │   ├── search_param.py   #    → LLM parameter extraction
│   │   ├── market_search.py  #    → Tavily search execution
│   │   ├── policy_rag.py     #    → Credit policy retrieval
│   │   ├── quotation.py      #    → Installment calculation
│   │   └── submission.py     #    → Supabase data write (Deprecated/Future)
│   ├── tools/                # 🛠️ Tool Implementations
│   │   ├── tavily_search.py  #    → Search API wrapper
│   │   ├── policy_retriever.py #  → Vector DB RAG tool
│   │   └── calculator.py     #    → Loan math tool
│   └── custom.py             # 🔌 DataRobot hooks
├── tests/                    # 🧪 Unit tests
└── Taskfile.yml              # 📋 Build & Run commands
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites
*   Python 3.9+
*   [uv](https://docs.astral.sh/uv/) (Dependency Manager)
*   [Taskfile](https://taskfile.dev/) (Command Runner)

### 2. Configure Environment
Create a `.env` file in the root directory:

```bash
cp .env.template .env
```

Populate the following secrets:

```bash
# DataRobot (Compute & LLM)
DATAROBOT_API_TOKEN=...
DATAROBOT_ENDPOINT=...

# Tools
TAVILY_API_KEY=tvly-...
DATAROBOT_VECTOR_DB_ID=...
```

### 3. Install Dependencies
```bash
task agent_langgraph:install
```

### 4. Create Database Tables
*(Automated via SQLite checkpointer, no manual setup required for local dev)*

---

## 🧪 Testing

### Local Development Server
Start the hot-reloading dev server:
```bash
task agent_langgraph:dev
```

Test with `curl` (simulating DataRobot runtime):
```bash
curl -X POST http://localhost:8842/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agent",
    "messages": [{"role": "user", "content": "I want a 2024 Hyundai Tucson"}],
    "extra_body": {"thread_id": "test-session-1"},
    "stream": false
  }'
```

### ChainLit UI (Interactive Playground)
For a chat interface similar to production:
```bash
task agent_langgraph:chainlit
```

### CLI Testing
Run one-off commands via the CLI:
```bash
task agent_langgraph:cli -- execute --user_prompt "Status of request 123"
```

---

## 📦 Deployment

Deploy to DataRobot Custom Models:

```bash
task deploy
```
This will containerize the agent, upload it to DataRobot, and deploy it as a prediction API.

---

## 📄 Documentation Links
*   [Product Requirements (PRD)](./PRD.md)
*   [Technical Design](./TECHNICAL_DESIGN.md)
*   [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
