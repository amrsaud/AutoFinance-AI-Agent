**Product Requirements Document (PRD)**

**Project Name:** AutoFinance AI Agent ("Financial Co-Pilot") – MVP

**Version:** 1.0

**Status:** Draft

**Target Market:** Egypt

## ---

**1\. Executive Summary & Business Context**

### **The Opportunity**

Buying a vehicle in emerging markets like Egypt is a fragmented, high-friction experience. Customers face a significant disconnect between **Discovery** (finding a car on disjointed listing sites) and **Transaction** (calculating affordability and securing a loan). This results in lost leads and customer frustration.

### **The Solution**

The AutoFinance AI Agent is an autonomous "Financial Co-Pilot" that bridges this gap. Unlike standard chatbots, it is an agentic system capable of:

* **Aggregating Market Data:** Real-time searching of fragmented marketplaces (e.g., Hatla2ee, Dubizzle) to find vehicles.  
* **Enforcing Policy:** Retrieving internal credit rules via Retrieval-Augmented Generation (RAG) to ensure quoted interest rates match the user's risk profile.  
* **Executing Transactions:** Calculating precise monthly installments and capturing high-intent applications for back-office review.

### **Business Value**

* **Reduced Turnaround Time:** Instant pre-approval estimates vs. days of manual processing.  
* **Higher Conversion:** Users are captured at the moment of intent (finding the car).  
* **Trust:** A transparent "Show Your Work" UI builds confidence in the financial advice.

## ---

**2\. MVP Scope & Constraints**

To ensure a rapid go-to-market strategy, the MVP will operate under specific constraints:

* **No Core Banking Integration (CBS):** The system will *not* connect to the live banking ledger or credit bureau.  
* **Back-Office Handoff:** All applications will be stored in an external database (**Supabase**) for manual review by the operations team.  
* **Lead Capture Focus:** The primary goal is to qualify the lead and the asset (vehicle) and provide a "Pre-Approval Estimate," not a binding contract.

## ---

**3\. Functional Requirements (FR)**

The user journey is divided into 5 distinct phases.

### **Phase 1: Onboarding & Routing**

* **FR-01 (Welcome & Instructions):**  
  * The Agent must initiate the session with a welcome message explaining its capability (finding cars \+ calculating loans).  
  * The Agent must present two clear options:  
    1. **Start New Request:** Begin the vehicle search journey.  
    2. **Check Status:** Query the status of a previous application using a **Request ID**.

### **Phase 2: Market Discovery & Validation**

* **FR-02 (Prompt Formalization):**  
  * The Agent must accept natural language input (e.g., "I want a 2021 Hyundai Tucson") and restructure it into a standardized search query (Make, Model, Year Range, Price Cap).  
* **FR-03 (Human-in-the-Loop Validation):**  
  * **Constraint:** Before executing the search, the Agent must display the formalized parameters to the user and ask for explicit confirmation (e.g., "I am about to search for a Hyundai Tucson, 2021 model or newer. Is this correct?").  
* **FR-04 (Market Search \- Tavily API):**  
  * Upon confirmation, the Agent must utilize the **Tavily Search API** to scour web listings (specifically targeting Egyptian marketplaces).  
  * The output must be formatted to show: Vehicle Name, Price, Year, Mileage, and a **Citation/Source Link** for every result.  
* **FR-05 (Selection & Loop):**  
  * The user must be able to select one specific vehicle from the results.  
  * If no result appeals to the user, the Agent must offer to restart the search or refine parameters.

### **Phase 3: Financial Profiling & Policy Logic**

* **FR-06 (Data Collection \- Financial):**  
  * Once a car is selected, the Agent must collect:  
    * Monthly Income (EGP).  
    * Employment Type (e.g., Salaried, Self-Employed, Corporate).  
* **FR-07 (Policy Retrieval \- RAG):**  
  * The system must maintain a **Vector Database** containing current internal Credit Policies, Risk Rules (Debt Burden Ratio), and Interest Rate grids.  
  * The Agent must query this database using the User’s Profile \+ Selected Car Details (Age/Price) to retrieve the correct rule set.  
* **FR-08 (Eligibility Check):**  
  * The Agent must cross-reference user data with retrieved rules *before* generating a quote.  
  * **Failure Scenario:** If no policy matches (e.g., car \> 10 years old, income below minimum), the Agent must inform the user politely and trigger a "Start Over" workflow.

### **Phase 4: Quotation & Confirmation**

* **FR-09 (Installment Calculation):**  
  * The Agent must calculate the precise monthly installment based on:  
    * Principal \= Car Price (from Tavily result).  
    * Interest Rate \= Derived from RAG Policy.  
    * Tenure \= Standard or User-selected (e.g., 60 months).  
* **FR-10 (Pre-Application Review):**  
  * The Agent must present a full summary card: *Vehicle Details \+ Total Price \+ Monthly Installment \+ Interest Rate.*  
  * The Agent must ask for confirmation: "Do you want to proceed with a formal request based on these terms?"

### **Phase 5: Submission & Persistence**

* **FR-11 (Lead Capture \- PII):**  
  * Upon confirmation, the Agent must collect personal contact details: Full Name, Email, Phone Number, National ID (optional).  
* **FR-12 (Data Storage \- Supabase):**  
  * The Agent must generate a unique **Request ID**.  
  * All data (User Profile, Vehicle Data, Financial Quote) must be formatted and stored in a **Supabase** table.  
  * Status must be set to "Pending Review."  
* **FR-13 (Completion):**  
  * The Agent must display the **Request ID** to the user.  
  * The Agent must provide instructions on how to check the status later (linking back to FR-01).

## ---

**4\. Technical Architecture (MVP Stack)**

* **Orchestration Engine:** LangChain (LangGraph for state management) with Datarobot.  
* **LLM:** DataRobot LLM Gateway 
* **Search Tool:** Tavily Search API (optimized for RAG/Agentic search).  
* **Database:** Supabase (PostgreSQL) for transactional data.  
* **Vector Database:** DataRobot VectorDB

## ---

