# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Search Parameter Node - LLM-powered parameter extraction.

Uses the LLM to extract structured search parameters from natural language
user input (e.g., "I want a 2021 Hyundai Tucson under 1 million").
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_litellm import ChatLiteLLM
from models import AgentState, SearchParams, WorkflowPhase

EXTRACT_PARAMS_SYSTEM_PROMPT = """You are a search parameter extraction assistant for a car financing service in Egypt.

Your task is to extract vehicle search parameters from the user's natural language input.

Extract the following fields (use null if not mentioned):
- make: Vehicle manufacturer (e.g., "Hyundai", "Toyota", "BMW")
- model: Vehicle model (e.g., "Tucson", "Corolla", "X5")
- year_min: Minimum model year (e.g., 2020)
- year_max: Maximum model year (e.g., 2023)
- price_cap: Maximum price in EGP (e.g., 1000000 for "under 1 million")

IMPORTANT:
- All prices should be in EGP (Egyptian Pounds)
- If user mentions "million" or "مليون", multiply by 1,000,000
- If only one year is mentioned, use it as year_min
- Common Arabic car terms: 
  - عربية/سيارة = car
  - موديل = model
  - سنة = year

Respond with a JSON object only, no additional text.
Example: {"make": "Hyundai", "model": "Tucson", "year_min": 2021, "year_max": null, "price_cap": 1500000}
"""


def search_param_node(state: AgentState, llm: ChatLiteLLM = None) -> dict[str, Any]:
    """
    Extract structured search parameters from user input using LLM.

    Returns a confirmation message asking the user to verify the extracted parameters.
    """
    # Get the last user message
    last_message = None
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_message = msg
            break

    if not last_message:
        return {
            "messages": [
                AIMessage(content="Please tell me what kind of car you're looking for.")
            ],
            "current_phase": WorkflowPhase.DISCOVERY,
        }

    user_input = str(last_message.content)

    # If LLM is provided, use it to extract parameters
    if llm:
        messages = [
            SystemMessage(content=EXTRACT_PARAMS_SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ]

        response = llm.invoke(messages)

        try:
            params_dict = json.loads(str(response.content))
            search_params = SearchParams(**params_dict)
        except (json.JSONDecodeError, ValueError):
            # Fallback: try basic extraction
            search_params = _basic_param_extraction(user_input)
    else:
        # No LLM available, use basic extraction
        search_params = _basic_param_extraction(user_input)

    # Build confirmation message
    confirmation_msg = _build_confirmation_message(search_params)

    return {
        "messages": [AIMessage(content=confirmation_msg)],
        "search_params": search_params,
        "search_params_confirmed": False,
        "awaiting_confirmation": "search",
        "current_phase": WorkflowPhase.DISCOVERY,
    }


def _basic_param_extraction(text: str) -> SearchParams:
    """
    Basic extraction without LLM - looks for common patterns.
    This is a fallback when LLM is not available.
    """
    import re

    text_lower = text.lower()

    # Common car makes
    makes = [
        "hyundai",
        "toyota",
        "honda",
        "bmw",
        "mercedes",
        "kia",
        "nissan",
        "mazda",
        "ford",
        "chevrolet",
        "volkswagen",
        "audi",
        "peugeot",
        "renault",
    ]
    make = None
    for m in makes:
        if m in text_lower:
            make = m.title()
            break

    # Common models - simplified list
    models = {
        "tucson": "Tucson",
        "elantra": "Elantra",
        "accent": "Accent",
        "corolla": "Corolla",
        "camry": "Camry",
        "rav4": "RAV4",
        "civic": "Civic",
        "accord": "Accord",
        "crv": "CR-V",
        "sportage": "Sportage",
        "cerato": "Cerato",
        "sunny": "Sunny",
        "sentra": "Sentra",
    }
    model = None
    for key, val in models.items():
        if key in text_lower:
            model = val
            break

    # Extract year
    year_match = re.search(r"20[0-3]\d", text)
    year_min = int(year_match.group()) if year_match else None

    # Extract price
    price_cap = None
    price_patterns = [
        (r"(\d+(?:\.\d+)?)\s*(?:million|مليون)", 1_000_000),
        (r"(\d{1,3}(?:,\d{3})+)", 1),
        (r"(\d{4,})", 1),
    ]
    for pattern, multiplier in price_patterns:
        match = re.search(pattern, text.replace(",", ""))
        if match:
            value = float(match.group(1).replace(",", ""))
            if multiplier > 1 or value > 10000:
                price_cap = value * multiplier
                break

    return SearchParams(
        make=make,
        model=model,
        year_min=year_min,
        year_max=None,
        price_cap=price_cap,
    )


def _build_confirmation_message(params: SearchParams) -> str:
    """Build a user-friendly confirmation message."""
    parts = []

    if params.make:
        parts.append(f"**Make:** {params.make}")
    if params.model:
        parts.append(f"**Model:** {params.model}")
    if params.year_min and params.year_max:
        parts.append(f"**Year Range:** {params.year_min} - {params.year_max}")
    elif params.year_min:
        parts.append(f"**Year:** {params.year_min} or newer")
    elif params.year_max:
        parts.append(f"**Year:** Up to {params.year_max}")
    if params.price_cap:
        parts.append(f"**Maximum Price:** {params.price_cap:,.0f} EGP")

    if not parts:
        return (
            "I couldn't extract specific search criteria from your message. "
            "Could you please tell me:\n"
            "- What **make** and **model** are you looking for?\n"
            "- What **year** or year range?\n"
            "- Do you have a **maximum budget**?"
        )

    criteria = "\n".join(f"- {part}" for part in parts)

    return f"""
## 🔍 Search Criteria Confirmation

I'll search Egyptian marketplaces for:

{criteria}

**Is this correct?** (Reply "Yes" to search, or tell me what to change)
"""
