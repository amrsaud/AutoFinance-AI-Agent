# Copyright 2025 DataRobot, Inc.
# Licensed under the Apache License, Version 2.0
"""Welcome node - greets users and guides them to search or status check."""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from datarobot_genai.core.agents import make_system_prompt

try:
    from state import AutoFinanceState
except ImportError:
    from ..state import AutoFinanceState


WELCOME_PROMPT = """You are AutoFinance AI, a helpful financial co-pilot for car buyers in Egypt.

Your role is to:
1. Welcome users warmly
2. Explain what you can do:
   - Search for vehicles across Egyptian marketplaces (Hatla2ee, Dubizzle, OLX)
   - Calculate loan eligibility and monthly installments
   - Submit pre-approval applications
3. Guide users to either:
   - Start a new vehicle search (ask what car they're looking for)
   - Check an existing application status (ask for their Request ID)

Be friendly, professional, and concise. Ask clarifying questions if needed.
Always respond in the same language the user uses (Arabic or English)."""


def create_welcome_node(llm, tools=None):
    """Create welcome node using create_react_agent.

    Args:
        llm: The LLM instance to use
        tools: Optional tools (typically none for welcome)
    """
    return create_react_agent(
        llm,
        tools=tools or [],
        prompt=make_system_prompt(WELCOME_PROMPT),
        name="Welcome Agent",
    )


async def welcome_node(
    state: AutoFinanceState, config: RunnableConfig = None, *, agent=None
) -> dict[str, Any]:
    """Welcome node handler.

    This is a wrapper that can be used directly or with a pre-created agent.
    """
    # If called with agent instance from the main AutoFinanceAgent class
    if agent:
        return await agent.ainvoke(state, config=config)

    # Fallback for direct invocation (shouldn't happen in normal flow)
    return {
        "messages": [],
        "current_phase": "onboarding",
    }
