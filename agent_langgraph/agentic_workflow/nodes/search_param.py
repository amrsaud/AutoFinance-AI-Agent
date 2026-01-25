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
Search parameter extraction node.
Extracts structured search parameters from natural language using LLM.
"""

import logging

from langchain_core.messages import AIMessage, SystemMessage
from opentelemetry import trace

from models import SearchParams

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

EXTRACTION_SYSTEM_PROMPT = """Extract car search parameters from the user message.
Extract the following fields:
- make: Car manufacturer (e.g., "Hyundai", "Toyota", "BMW")
- model: Car model (e.g., "Tucson", "Camry", "X5")
- year_min: Minimum year filter (e.g., 2023)
- year_max: Maximum year filter (e.g., 2024)
- price_max: Maximum price in EGP (e.g., 500000)

Set raw_query to the original user message.
If a field is not mentioned, leave it as null."""


async def extract_search_params(state: dict, llm) -> dict:
    """Extract search parameters from natural language using LLM structured output.

    Args:
        state: The current agent state containing messages.
        llm: The language model to use for extraction.

    Returns:
        dict: Updated state with search_params and confirmation message.
    """
    with tracer.start_as_current_span("extract_search_params") as span:
        messages = state.get("messages", [])
        if not messages:
            return {
                "messages": [
                    AIMessage(content="Please tell me what car you're looking for.")
                ]
            }

        last_message = messages[-1].content
        span.set_attribute("node.input", last_message)

        # Use LLM with structured output
        structured_llm = llm.with_structured_output(SearchParams)
        params = await structured_llm.ainvoke(
            [SystemMessage(content=EXTRACTION_SYSTEM_PROMPT), messages[-1]]
        )
        params.raw_query = last_message

        span.set_attribute("node.output.make", params.make or "")
        span.set_attribute("node.output.model", params.model or "")

        # Build confirmation message
        confirmation_parts = []
        if params.make:
            confirmation_parts.append(f"**{params.make}**")
        else:
            confirmation_parts.append("**Any make**")

        if params.model:
            confirmation_parts.append(f"**{params.model}**")
        else:
            confirmation_parts.append("**Any model**")

        confirmation_msg = f"I'll search for: {' '.join(confirmation_parts)}"

        if params.year_min:
            confirmation_msg += f" from **{params.year_min}**"
        if params.year_max:
            confirmation_msg += f" to **{params.year_max}**"
        if params.price_max:
            confirmation_msg += f" under **{params.price_max:,} EGP**"

        confirmation_msg += "\n\n✅ Should I proceed with the search? (yes/no)"

        logger.info(f"Extracted params: make={params.make}, model={params.model}")

        return {
            "search_params": params,
            "messages": [AIMessage(content=confirmation_msg)],
        }
