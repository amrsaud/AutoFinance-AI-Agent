# Copyright 2025 DataRobot, Inc.
# Licensed under the Apache License, Version 2.0
"""
Chainlit integration for AutoFinance AI Agent.
"""

import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage
from langchain.schema.runnable.config import RunnableConfig

from agentic_workflow.agent import AutoFinanceAgent
from agentic_workflow.config import Config

config = Config()


def get_initial_state() -> dict:
    """Create initial state for the agent."""
    return {
        "messages": [],
        "current_phase": "onboarding",
        "search_params": None,
        "search_results": None,
        "selected_vehicle": None,
        "monthly_income": None,
        "employment_type": None,
        "applicable_policy": None,
        "financial_quote": None,
        "customer_info": None,
        "request_id": None,
    }


@cl.on_chat_start
async def start_chat() -> None:
    """Initialize chat session with state."""
    cl.user_session.set("state", get_initial_state())

    # Initialize agent once
    agent = AutoFinanceAgent(
        messages=[],
        model=config.llm_default_model,
        stream=True,
    )
    # Compile graph once
    graph = agent.workflow.compile()

    cl.user_session.set("graph", graph)

    await cl.Message(
        content="""🚗 **Welcome to AutoFinance AI!**

I help you find cars in Egypt and calculate loan options instantly.

**What would you like to do?**
1️⃣ **Start New Request** - "Find me a Toyota Corolla"
2️⃣ **Check Status** - Provide your Request ID (AF-XXXXXX-XXXX)"""
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle messages with LangGraph streaming and Chainlit callbacks."""
    graph = cl.user_session.get("graph")
    state = cl.user_session.get("state")

    if not state:
        state = get_initial_state()

    # Append user message
    state["messages"].append(HumanMessage(content=message.content))

    # Config with callback handler for UI visualization
    cb = cl.LangchainCallbackHandler()
    runnable_config = RunnableConfig(
        callbacks=[cb], configurable={"thread_id": cl.context.session.id}
    )

    # Clean final answer container
    final_answer = cl.Message(content="")

    # Stream the graph execution
    # We use astream to get events/updates
    async for output in graph.astream(state, config=runnable_config):
        # Output is a dict of node_name: state_update
        for node_name, state_update in output.items():
            if node_name == "router":
                # Update our local state with the result from the router
                for key, value in state_update.items():
                    if value is not None:
                        state[key] = value

                # --- UI VISUALIZATION OF STATE CHANGES ---

                # 1. Phase Change
                if "current_phase" in state_update:
                    phase = state_update["current_phase"]
                    async with cl.Step(name="🔄 Phase Transition", type="run") as s:
                        s.output = f"Switched to: **{phase.upper()}**"

                # 2. Search Parameters (Readable)
                if "search_params" in state_update and state_update["search_params"]:
                    p = state_update["search_params"]
                    # Handle both dict and object
                    if hasattr(p, "dict"):
                        p = p.dict()
                    elif hasattr(p, "model_dump"):
                        p = p.model_dump()

                    async with cl.Step(name="🔍 Extracted Criteria", type="run") as s:
                        s.output = f"""**Make:** {p.get("make")}
**Model:** {p.get("model")}
**Years:** {p.get("year_from")} - {p.get("year_to")}
**Price Cap:** {p.get("price_cap", "None")}"""

                # 3. Selected Vehicle (Readable)
                if (
                    "selected_vehicle" in state_update
                    and state_update["selected_vehicle"]
                ):
                    v = state_update["selected_vehicle"]
                    if hasattr(v, "model_dump"):
                        v = v.model_dump()

                    async with cl.Step(name="✅ Vehicle Selected", type="run") as s:
                        s.output = f"""**{v.get("year")} {v.get("make")} {v.get("model")}**
**Price:** {v.get("price"):,.0f} EGP
**Mileage:** {v.get("mileage")} km
**Source:** {v.get("source_name")}"""

                # 4. Financial Quote (Readable)
                if (
                    "financial_quote" in state_update
                    and state_update["financial_quote"]
                ):
                    q = state_update["financial_quote"]
                    if hasattr(q, "model_dump"):
                        q = q.model_dump()

                    async with cl.Step(name="💰 Loan Calculation", type="run") as s:
                        s.output = f"""**Monthly Installment:** {q.get("monthly_installment"):,.0f} EGP
**Interest Rate:** {q.get("interest_rate")}%
**Tenure:** {q.get("tenure_months")} months
**Total Payment:** {q.get("total_payment"):,.0f} EGP"""

                # -----------------------------------------

                # Check if there are new messages to display
                if "messages" in state_update:
                    msgs = state_update["messages"]
                    for msg in msgs:
                        # We only want to display AI messages that are new
                        if isinstance(msg, AIMessage) and msg.content:
                            await final_answer.stream_token(msg.content)

    # Send the final aggregated message
    await final_answer.send()

    # Save state back to session
    cl.user_session.set("state", state)
