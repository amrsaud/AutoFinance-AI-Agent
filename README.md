# **Project Manifest: AutoFinance AI Agent**

### ***Intelligent Car Discovery & Financing Assistant***

## **1\. Executive Summary (Business Context)**

The Opportunity:  
Buying a vehicle in emerging markets like Egypt is a fragmented, high-friction experience. Customers face a disconnect between discovery (finding a car on disjointed listing sites) and transaction (calculating affordability and securing a loan).  
The Solution:  
The AutoFinance AI Agent is an autonomous "Financial Co-Pilot" that bridges this gap. Unlike standard chatbots, it is an agentic system capable of:

1. **Aggregating Market Data:** Real-time searching of fragmented marketplaces (e.g., Hatla2ee, Dubizzle) to find vehicles.  
2. **Enforcing Policy:** Retrieving internal credit rules (RAG) to ensure quoted interest rates match the user's risk profile.  
3. **Executing Transactions:** Calculating precise monthly installments and submitting formal loan applications to the core banking system.

**Business Value:**

* **Reduced Turnaround Time:** Instant pre-approval estimates vs. days of manual processing.  
* **Higher Conversion:** Users are captured at the moment of intent (finding the car).  
* **Trust:** Transparent "Show Your Work" UI builds confidence in the financial advice.

---

## **2\. Documentation**

### **Core Documentation**

| Document | Description |
|----------|-------------|
| [Product Requirements Document](product-requirements-document.md) | MVP scope, functional requirements, and user journey phases |

### **Agent Documentation**

| Document | Description |
|----------|-------------|
| [Architecture Overview](agent_langgraph/ARCHITECTURE.md) | System architecture, component structure, tools, database schema, and environment setup |
| [Agent LangGraph README](agent_langgraph/README.md) | DataRobot agent template information |

### **Infrastructure & Deployment**

| Document | Description |
|----------|-------------|
| [Infrastructure Configurations](infra/configurations/README.md) | Runtime configuration options and environment variable controls |
| [Feature Flags](infra/feature_flags/README.md) | Required DataRobot feature flags for deployment |

---

## **3\. DataRobot Template Documentation**

This project is built on the [DataRobot Agentic Workflow Templates](https://github.com/datarobot-community/datarobot-agent-templates). For comprehensive guides, refer to the official documentation:

### **Getting Started**

| Guide | Description |
|-------|-------------|
| [Installation](https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-install.html) | Prerequisites and setup instructions |
| [Quickstart](https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-quickstart.html) | Quick start guide to build and deploy agents |
| [Agent Components](https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-overview.html) | Overview of agent architecture and components |

### **Development**

| Guide | Description |
|-------|-------------|
| [Agent Authentication](https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-authentication.html) | Authentication setup for agents |
| [Use the Agent CLI](https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-cli-guide.html) | Command-line interface guide |
| [Customize Agents](https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-development.html) | How to customize agent workflows |
| [Add Python Packages](https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-python-packages.html) | Adding dependencies to your agent |
| [Configure LLM Providers (Code)](https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-llm-providers.html) | Configure LLM providers in code |
| [Configure LLM Providers (Metadata)](https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-llm-providers-metadata.html) | Configure LLM providers with metadata |

### **Tools & Tracing**

| Guide | Description |
|-------|-------------|
| [Deploy Agentic Tools](https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-tools.html) | How to deploy tools for agents |
| [Add Tools to Agents](https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-tools-integrate.html) | Integrate tools into agent workflows |
| [Implement Tracing](https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-tracing.html) | Add observability and tracing |
| [Access Request Headers](https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-request-headers.html) | Access HTTP headers in agents |

### **Debugging & Maintenance**

| Guide | Description |
|-------|-------------|
| [Debug in PyCharm](https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-debugging-pycharm.html) | Debugging agents in PyCharm |
| [Debug in VS Code](https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-debugging-vscode.html) | Debugging agents in VS Code |
| [Update Templates](https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-updating.html) | Updating agentic templates |
| [Troubleshooting](https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/agentic-troubleshooting.html) | Common issues and solutions |

### **Evaluation & Deployment**

| Guide | Description |
|-------|-------------|
| [Connect to a Playground](https://docs.datarobot.com/en/docs/agentic-ai/agentic-eval/agentic-playground.html) | Testing agents in playground |
| [Chat with Agents](https://docs.datarobot.com/en/docs/agentic-ai/agentic-eval/agentic-chatting.html) | Interacting with deployed agents |
| [Evaluate Metrics](https://docs.datarobot.com/en/docs/agentic-ai/agentic-eval/agentic-evaluation-tools.html) | Metrics and evaluation tools |
| [Review Tracing](https://docs.datarobot.com/en/docs/agentic-ai/agentic-eval/agentic-tracing.html) | Review agent traces |
| [Build Workflows](https://docs.datarobot.com/en/docs/agentic-ai/agentic-eval/agentic-workflow-build.html) | Building agentic workflows |
| [Deploy Workflows](https://docs.datarobot.com/en/docs/agentic-ai/agentic-eval/agentic-workflow-reg-deploy.html) | Register and deploy workflows |

### **How-To Guides**

| Guide | Description |
|-------|-------------|
| [Modify the CrewAI Template](https://docs.datarobot.com/en/docs/agentic-ai/agentic-walkthroughs/agentic-custom-workflow.html) | Customizing the CrewAI template |
| [Modify with AI Assistance](https://docs.datarobot.com/en/docs/agentic-ai/agentic-walkthroughs/agentic-custom-workflow-migrate.html) | Using AI to modify templates |
| [Add Data Registry Tools](https://docs.datarobot.com/en/docs/agentic-ai/agentic-walkthroughs/agentic-crewai-data-registry.html) | Integrating Data Registry tools |
| [Agentic Glossary](https://docs.datarobot.com/en/docs/agentic-ai/agentic-glossary.html) | Terminology and definitions |
