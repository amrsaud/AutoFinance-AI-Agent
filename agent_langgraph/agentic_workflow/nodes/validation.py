# Copyright 2025 DataRobot, Inc.
# Licensed under the Apache License, Version 2.0
"""Validation node - confirms search parameters before executing search."""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

try:
    from state import AutoFinanceState
except ImportError:
    from ..state import AutoFinanceState


async def validation_node(
    state: AutoFinanceState, config: RunnableConfig = None
) -> dict[str, Any]:
    """Validate/confirm search parameters before executing search.

    Returns:
        - If confirmed: transition to execute_search
        - If modification requested: transition back to parse_search
    """
    messages = state.get("messages", [])
    params = state.get("search_params")

    # Get last user message
    last_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_msg = msg.content.lower()
            break

    if not params:
        return {
            "messages": [AIMessage(content="What car are you looking for?")],
            "current_phase": "onboarding",
            "validation_result": "no_params",
        }

    # Check for confirmation
    confirmation_words = [
        "yes",
        "correct",
        "ok",
        "search",
        "proceed",
        "نعم",
        "صح",
        "تمام",
    ]
    if any(w in last_msg for w in confirmation_words):
        return {
            "current_phase": "execute_search",
            "validation_result": "confirmed",
        }

    # User wants to modify - they'll provide new criteria
    return {
        "current_phase": "parse_search",
        "validation_result": "modify",
    }
