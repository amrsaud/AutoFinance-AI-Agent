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
Market search node.
Executes Tavily search and parses results into Vehicle objects using LLM.
"""

import logging
from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from opentelemetry import trace

from models import Vehicle
from tools.tavily_search import search_egyptian_cars

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

PARSING_SYSTEM_PROMPT = """Extract vehicle listings from the search results.
For each listing, extract:
- make: Car manufacturer
- model: Car model
- year: Manufacturing year (if available)
- price: Price in EGP (if available)
- mileage: Mileage in kilometers (if available)
- location: Location in Egypt (if available)
- source: Must be 'hatla2ee' or 'dubizzle'
- source_url: Full URL to the vehicle listing

Return as a list of Vehicle objects.
Only include listings with valid URLs from hatla2ee.com or dubizzle.com.eg.
If you cannot find any valid listings, return an empty list."""


async def search_market(state: dict, llm) -> dict:
    """Execute Tavily search using constructed query and parse results with LLM.

    Args:
        state: The current agent state with search_params.
        llm: The language model to use for parsing.

    Returns:
        dict: Updated state with search_results and response message.
    """
    with tracer.start_as_current_span("market_search") as span:
        params = state.get("search_params")
        if not params:
            return {
                "messages": [
                    AIMessage(
                        content="No search parameters found. Please tell me what car you're looking for."
                    )
                ]
            }

        # Build consistent search query from extracted parameters
        query = params.build_search_query()
        span.set_attribute("node.input.constructed_query", query)

        logger.info(f"Executing search with constructed query: {query}")

        # Execute Tavily search with constructed query
        raw_results = search_egyptian_cars.invoke(query)

        # Use LLM to extract structured Vehicle objects
        try:
            structured_llm = llm.with_structured_output(List[Vehicle])
            vehicles = await structured_llm.ainvoke(
                [
                    SystemMessage(content=PARSING_SYSTEM_PROMPT),
                    HumanMessage(content=raw_results),
                ]
            )
        except Exception as e:
            logger.error(f"Error parsing vehicles: {e}")
            vehicles = []

        vehicle_count = len(vehicles) if vehicles else 0
        span.set_attribute("node.output.vehicle_count", vehicle_count)

        # Format response
        if vehicles:
            response = f"🚗 Found **{vehicle_count}** vehicles:\n\n"
            for i, v in enumerate(vehicles, 1):
                year_str = str(v.year) if v.year else "N/A"
                response += f"**{i}. {year_str} {v.make} {v.model}**\n"
                if v.price:
                    response += f"   💰 Price: {v.price:,} EGP\n"
                if v.mileage:
                    response += f"   📏 Mileage: {v.mileage:,} km\n"
                if v.location:
                    response += f"   📍 Location: {v.location}\n"
                response += f"   🔗 [{v.source}]({v.source_url})\n\n"
            response += "Would you like more details about any of these vehicles?"
        else:
            response = (
                "❌ No vehicles found matching your criteria.\n\n"
                "Try adjusting your search parameters or searching for a different car."
            )

        logger.info(f"Found {vehicle_count} vehicles")

        return {
            "search_results": vehicles or [],
            "search_confirmed": True,
            "messages": [AIMessage(content=response)],
        }
