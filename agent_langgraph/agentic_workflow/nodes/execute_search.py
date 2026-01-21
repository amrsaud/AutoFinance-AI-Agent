# Copyright 2025 DataRobot, Inc.
# Licensed under the Apache License, Version 2.0
"""Execute search node - runs Tavily search and parses results with LLM."""

import json
import re
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from datarobot_genai.core.agents import make_system_prompt

try:
    from state import AutoFinanceState, Vehicle, SearchParams
    from tools.tavily_search import search_vehicles
except ImportError:
    from ..state import AutoFinanceState, Vehicle, SearchParams
    from ..tools.tavily_search import search_vehicles


PARSE_RESULTS_PROMPT = """You are a data extractor for a car finding agent.

Extract vehicle details from search results and return them as a JSON list.

Rules:
1. Extract ONLY vehicles that match the requested make/model
2. Return a JSON array of objects with these fields:
   - make: string
   - model: string  
   - year: integer
   - price: float (in EGP, remove commas)
   - mileage: integer or null (in km)
   - source_name: string (Hatla2ee, Dubizzle, OLX, etc)
   - source_url: string (the URL to the listing)

3. Ignore results that are NOT the specific car requested
4. Maximum 5 vehicles

Return ONLY valid JSON, no explanation."""


def create_execute_search_node(llm, tools=None):
    """Create execute_search node using create_react_agent."""
    return create_react_agent(
        llm,
        tools=tools or [search_vehicles],
        prompt=make_system_prompt(PARSE_RESULTS_PROMPT),
        name="Search Executor Agent",
    )


async def execute_search_node(
    state: AutoFinanceState, config: RunnableConfig = None, *, llm=None
) -> dict[str, Any]:
    """Execute vehicle search and parse results using LLM."""
    params = state.get("search_params")

    if not params:
        return {
            "messages": [AIMessage(content="What car are you looking for?")],
            "current_phase": "onboarding",
        }

    # Handle both Pydantic object and dict
    if hasattr(params, "model_dump"):
        params_dict = params.model_dump()
    elif isinstance(params, dict):
        params_dict = params
    else:
        params_dict = dict(params)

    # Execute search
    raw_results = search_vehicles.invoke({"params": params_dict}, config=config)

    if not raw_results:
        return {
            "messages": [
                AIMessage(
                    content="No vehicles found. Try different criteria?\nExample: 'Toyota Corolla 2020-2024'"
                )
            ],
            "search_results": [],
            "current_phase": "onboarding",
        }

    # Parse with LLM
    vehicles = await _parse_vehicles_with_llm(raw_results, params, llm)

    if not vehicles:
        return {
            "messages": [
                AIMessage(
                    content="Found results but none matched your exact criteria. Try broadening your search."
                )
            ],
            "search_results": [],
            "current_phase": "onboarding",
        }

    # Format results message
    results_msg = f"📊 **Found {len(vehicles)} vehicles:**\n\n"
    for i, v in enumerate(vehicles, 1):
        results_msg += f"**{i}. {v.year} {v.make} {v.model}** - {v.price:,.0f} EGP\n"
        mileage_info = f"{v.mileage:,} km" if v.mileage else "N/A"
        if v.source_url:
            results_msg += f"   📍 {mileage_info} | [{v.source_name}]({v.source_url})\n"
        else:
            results_msg += f"   📍 {mileage_info} | {v.source_name}\n"

    results_msg += f"\n**Which one interests you?** (Enter 1-{len(vehicles)})"

    return {
        "messages": [AIMessage(content=results_msg)],
        "search_results": vehicles,
        "current_phase": "selection",
    }


async def _parse_vehicles_with_llm(
    results: list[dict], params: SearchParams, llm
) -> list[Vehicle]:
    """Use LLM to parse raw search results into Vehicle objects."""
    if not llm:
        return []

    # Format snippets for LLM
    snippets = []
    for i, res in enumerate(results):
        snippets.append(
            f"Result {i + 1}:\nTitle: {res.get('title')}\nURL: {res.get('url')}\nContent: {res.get('content')}"
        )

    context = "\n\n".join(snippets)

    # Handle both Pydantic and dict
    if hasattr(params, "make"):
        make, model = params.make, params.model
        year_from, year_to = params.year_from, params.year_to
        price_cap = params.price_cap
    else:
        make, model = params.get("make"), params.get("model")
        year_from, year_to = params.get("year_from"), params.get("year_to")
        price_cap = params.get("price_cap")

    prompt = f"""Extract vehicle details from these search results for: {make} {model} {year_from}-{year_to}

Search Results:
{context}

Return ONLY valid JSON list (max 5 vehicles):
[{{"make": "...", "model": "...", "year": 2022, "price": 500000, "mileage": 50000, "source_name": "...", "source_url": "..."}}]"""

    try:
        response = await llm.ainvoke(prompt)
        content = response.content.strip()

        # Clean markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        data = json.loads(content)
        vehicles = []

        for item in data[:5]:  # Max 5 vehicles
            v = Vehicle(
                make=item.get("make", make),
                model=item.get("model", model),
                year=item.get("year", year_from),
                price=float(item.get("price", 0)),
                mileage=item.get("mileage"),
                source_url=item.get("source_url", ""),
                source_name=item.get("source_name", "Unknown"),
            )
            # Validation
            if v.price > 0:
                if not price_cap or v.price <= price_cap:
                    vehicles.append(v)

        return vehicles
    except Exception as e:
        print(f"LLM parsing error: {e}")
        return []
