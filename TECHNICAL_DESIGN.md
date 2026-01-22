# **Technical Design Document: AutoFinance AI Agent (MVP)**

**Project Name:** AutoFinance AI Agent ("Financial Co-Pilot")

**Version:** 1.0

**Target Platform:** DataRobot Custom Models

**Orchestration:** LangGraph

**Persistence:** Supabase (PostgreSQL)

**Status:** Approved for Implementation

## ---

**1\. Executive Summary**

This document defines the technical architecture for the **AutoFinance AI Agent**, a stateful conversational system designed to assist users in the Egyptian market with finding vehicles and calculating loan affordability.

The system addresses the challenge of running a long-running, multi-turn application within **DataRobot's stateless Custom Model environment**. It utilizes a **Sidecar Persistence Pattern**, where DataRobot handles compute and inference, while Supabase manages conversational memory and transactional state. This ensures that the agent can "remember" user context (e.g., selected car, income) across days or weeks, despite the ephemeral nature of the model container.

## ---

**2\. System Architecture**

### **2.1 High-Level Design**

The solution follows a **Hub-and-Spoke** architecture centered around the DataRobot Custom Model runner.

1. **Compute Layer (DataRobot):**  
   * **Custom Model Container:** Python 3.9+ runtime hosting the LangGraph orchestration engine.  
   * **Concurrency Model:** Implements an **Asynchronous Event Loop** nested within a ThreadPoolExecutor (max 1 worker) to isolate the agent's I/O operations from the main DataRobot thread.  
   * **LLM Gateway:** Connects to DataRobot's LLM Gateway for reasoning and generation.  
2. **Persistence Layer (Supabase/PostgreSQL):**  
   * **Checkpointer:** Stores the serialized LangGraph state (the "save game" file) keyed by the Session ID.  
   * **Business Ledger:** Stores structured application data (leads, quotes) for back-office review.  
3. **Integration Layer:**  
   * **Search:** **Tavily Search API** for retrieving real-time vehicle listings from Egyptian marketplaces (Hatla2ee, Dubizzle).  
   * **Knowledge Base:** **DataRobot Vector Database** for RAG-based retrieval of credit policies and interest rate grids.

### **2.2 Data Flow**

1. **Ingest:** User sends a message via the Frontend. DataRobot receives the request containing the prompt and association\_id (Session ID).  
2. **Hydrate:** The Agent uses the association\_id to query Supabase for the existing thread state.  
3. **Reason:** LangGraph processes the input, potentially calling the LLM Gateway or Tavily API.  
4. **Action:** The Agent updates the state (e.g., "Income Collected").  
5. **Persist:** The new state is serialized and upserted back to Supabase.  
6. **Respond:** The text response is returned to the user; the container becomes idle.

## ---

**3\. Data Architecture**

### **3.1 Runtime Agent State**

The agent maintains a structured state object during execution. This schema defines the "Short-Term Memory" of the bot.

| Field | Type | Description |
| :---- | :---- | :---- |
| **messages** | List | The append-only log of all Human and AI messages. |
| **current\_phase** | String | Tracks the macro-status (e.g., discovery, profiling, quotation). |
| **search\_params** | Object | Structured query extracted from user input (Make, Model, Year Range, Price Cap). |
| **search\_results** | List | List of Vehicle objects returned by Tavily (Make, Model, Price, Mileage, Source URL). |
| **selected\_vehicle** | Object | The single Vehicle object the user has chosen to finance. |
| **monthly\_income** | Float | User's self-reported monthly income (EGP). |
| **employment\_type** | String | User's employment category (Corporate, Self-Employed). |
| **applicable\_policy** | Object | The specific Credit Policy retrieved via RAG (Rate, Max Tenure, Eligibility Boolean). |
| **financial\_quote** | Object | Calculated loan details (Principal, Monthly Installment, Total Interest). |
| **customer\_info** | Object | PII collected at submission (Name, Email, Phone, National ID). |
| **request\_id** | String | Unique UUID generated upon successful submission. |

### **3.2 Persistent Database Schema (Supabase)**

Two primary schemas are required in the PostgreSQL database.

#### **Table A: checkpoints (Managed by LangGraph)**

*   **Implementation:** Uses the official `langgraph-checkpoint-postgres` library.
*   **Automation:** The table is automatically created via `checkpointer.setup()` on agent startup.
*   **Purpose:** Stores the binary state of the conversation graph (checkpoints and writes).
*   **Key Columns:** thread\_id (PK), checkpoint (Blob), metadata (JSONB).

#### **Table B: applications (Business Data)**

* **Purpose:** Stores the final "Lead" for the operations team.  
* **Columns:**  
  * request\_id (UUID, Primary Key)  
  * session\_id (Text, Foreign Key to Thread)  
  * user\_name (Text)  
  * contact\_details (JSONB \- Phone, Email, National ID)  
  * vehicle\_summary (JSONB \- Make, Model, Price, Link)  
  * financial\_summary (JSONB \- Income, Installment, Rate)  
  * status (Text \- Default: 'Pending Review')  
  * created\_at (Timestamp)

## ---

**4\. Application Logic (LangGraph Design)**

The orchestration engine uses a **StateGraph** with a **"Check-Ask-Exit"** loop pattern. The graph does not pause; it halts execution until a new user message triggers a resumption.

### **4.1 Node Definitions**

1. **RouterNode (Entry Point):**  
   * Analyzes the incoming user message and current state.  
   * Routes to SearchParamNode if the user is new.  
   * Routes to StatusCheckNode if the user asks for an update.  
   * Passes through to the active phase if a conversation is in progress.  
2. **SearchParamNode (LLM Reasoning):**  
   * **Input:** Raw user text (e.g., "I want a 2020 Kia Sportage").  
   * **Action:** Uses DataRobot LLM to extract structured SearchParams.  
   * **Output:** Updates search\_params in state. Returns a confirmation message to the user.  
3. **MarketSearchNode (Tool Execution):**  
   * **Trigger:** Only runs after explicit user confirmation ("Yes").  
   * **Action:** Calls Tavily API with a query optimized for Egyptian sites (site:hatla2ee.com OR site:dubizzle.com.eg).  
   * **Output:** Updates search\_results with a list of Vehicle objects.  
4. **ProfilingLogicNode (The Data Loop):**  
   * **Action:** This is a pure logic node (no LLM). It checks the state for monthly\_income and employment\_type.  
   * **Routing:**  
     * If monthly\_income is NULL \-\> Routes to AskIncomeNode.  
     * If employment\_type is NULL \-\> Routes to AskEmploymentNode.  
     * If both exist \-\> Routes to PolicyRAGNode.  
5. **PolicyRAGNode (Retrieval):**  
   * **Action:** Queries the DataRobot Vector Database using the User Profile \+ Vehicle Age.  
   * **Output:** Retrieves CreditPolicy (Interest Rate, DBR Limit). Performs an eligibility check (e.g., is Vehicle Age \> 10 years?).  
6. **QuotationNode (Deterministic Math):**  
   * **Action:** Calculates the monthly installment using the standard PMT formula.  
   * **Constraint:** strictly Python-based math; no LLM usage for calculations.  
   * **Output:** Updates financial\_quote and presents the summary card to the user.  
7. **SubmissionNode (Write):**  
   * **Action:** Generates request\_id, writes the full record to the applications table in Supabase.  
   * **Output:** Returns the ID to the user and marks the workflow as complete.

### **4.2 Critical Logic Flows**

#### **The "Human-in-the-Loop" Validation (FR-03 & FR-10)**

* **Mechanism:** interrupt\_before Logic.  
* **Implementation:** The graph is configured to strictly halt before entering the MarketSearchNode and SubmissionNode.  
* **Resume Condition:** The system requires a specific "Yes/Confirm" intent from the user in the next turn to proceed past the interrupt.

#### **The "Data Collection" Loop (FR-06)**

* **Mechanism:** Cyclic Conditional Edges.  
* **Logic:** The graph edges form a loop. The ProfilingLogicNode will repeatedly route the user back to question nodes (AskIncome, AskEmployment) until the state requirements are met. The graph cannot physically progress to the QuotationNode until the data fields are non-null.

## ---

**5\. Interface Specifications**

### **5.1 DataRobot LLM Gateway**

* **Model:** Fast reasoning model (e.g., GPT-4o or equivalent hosted model).  
* **Temperature:** 0.0 (Strict determinism required for JSON extraction).  
* **System Prompt:** Must include Egyptian market context (currency \= EGP) and specific instruction to ignore non-automotive queries.

### **5.2 Tavily Search API**

* **Query Formatting:** Queries must be suffixed with specific domain constraints to ensure relevance.  
  * *Template:* "{Make} {Model} {Year} price in Egypt site:hatla2ee.com OR site:dubizzle.com.eg"  
* **Result Parsing:** The implementation must parse the raw text snippets to extract specific "Price" integers for calculations.

### **5.3 Supabase (PostgreSQL)**

* **Connection:** Must use the **Transaction Pooler** (Port 6543\) to handle the ephemeral connections from DataRobot containers efficiently.  
* **Security:** Row Level Security (RLS) is not required for the MVP backend access, but the database credentials must be secured in DataRobot Environment Variables.

## ---

**6\. Implementation Guidelines for Engineering**

1. **Thread Safety:** The load\_model function in custom.py must initialize the ThreadPoolExecutor(max\_workers=1). Do not increase worker count; concurrency is handled by DataRobot scaling replicas, not threads.  
2. **State Schema:** Implement the Pydantic models exactly as defined in the Data Architecture section to ensure type safety between the LLM and the Database.  
3. **Error Handling:**  
   * If Tavily returns 0 results, the agent must not crash. It should transition to a "Refinement Node" that asks the user to broaden their search (e.g., "I couldn't find a 2022 model, but I found 2021\. Interested?").  
   * If RAG retrieval fails, default to a conservative "Fallback Policy" (e.g., generic interest rate) but flag the quote as "Estimate Only."  
4. **Zombie Prevention:** In the Router Node, always check if the request\_id is already populated. If the user says "Thanks" after submission, do not try to parse "Thanks" as a car model. Reply with a standard closing intent.