# Copyright 2025 DataRobot, Inc.
# Licensed under the Apache License, Version 2.0
"""Parse search node - extracts structured search params from user query."""

import json
import re
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from datarobot_genai.core.agents import make_system_prompt

try:
    from state import AutoFinanceState, SearchParams
except ImportError:
    from ..state import AutoFinanceState, SearchParams


PARSE_SEARCH_PROMPT = """You are a search parameter extractor for a car finding agent in Egypt.

Your task is to extract structured search parameters from user messages.

When the user describes what car they want, extract:
- make: Vehicle manufacturer (e.g., Toyota, Hyundai, BMW)
- model: Vehicle model (e.g., Corolla, Tucson, X5)
- year_from: Minimum year (default: 5 years ago if not specified)
- year_to: Maximum year (default: current year)
- price_cap: Maximum price in EGP (null if not specified)

Respond with ONLY a JSON object:
{"make": "Toyota", "model": "Corolla", "year_from": 2020, "year_to": 2025, "price_cap": null}

Handle Arabic and English inputs. Be flexible with spelling variations."""


def create_parse_search_node(llm, tools=None):
    """Create parse_search node using create_react_agent."""
    return create_react_agent(
        llm,
        tools=tools or [],
        prompt=make_system_prompt(PARSE_SEARCH_PROMPT),
        name="Parse Search Agent",
    )


async def parse_search_node(
    state: AutoFinanceState, config: RunnableConfig = None, *, llm=None
) -> dict[str, Any]:
    """Extract search parameters from user message using LLM."""
    messages = state.get("messages", [])

    # Get last user message
    last_msg = None
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            last_msg = msg.content
            break
        elif hasattr(msg, "content") and not hasattr(msg, "type"):
            last_msg = msg.content
            break

    if not last_msg or not llm:
        return {
            "messages": [
                AIMessage(
                    content="Please tell me what car you're looking for.\nExample: 'Find me a 2022 Toyota Corolla under 500,000 EGP'"
                )
            ],
            "current_phase": "onboarding",
        }

    prompt = f"""Extract car search parameters from: "{last_msg}"
Return ONLY a JSON object with: make, model, year_from, year_to, price_cap
Example: {{"make":"Toyota","model":"Corolla","year_from":2020,"year_to":2025,"price_cap":null}}"""

    try:
        response = await llm.ainvoke(prompt)
        match = re.search(r"\{[^}]+\}", response.content)
        if match:
            params = json.loads(match.group(0))
            search_params = SearchParams(**params)

            return {
                "messages": [
                    AIMessage(
                        content=f"""🔍 **Search Parameters**

- **Make:** {search_params.make}
- **Model:** {search_params.model}
- **Years:** {search_params.year_from} - {search_params.year_to}
- **Max Price:** {f"{search_params.price_cap:,.0f} EGP" if search_params.price_cap else "No limit"}

**Is this correct?** (Yes to search, or tell me what to change)"""
                    )
                ],
                "search_params": search_params,
                "current_phase": "validation",
            }
    except Exception as e:
        print(f"Parse error: {e}")

    return {
        "messages": [
            AIMessage(
                content="I couldn't understand that. Please tell me what car you're looking for.\nExample: 'Find me a 2022 Toyota Corolla under 500,000 EGP'"
            )
        ],
        "current_phase": "onboarding",
    }
