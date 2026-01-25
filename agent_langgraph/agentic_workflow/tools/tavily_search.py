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
Tavily search tool for Egyptian car marketplaces.
Searches hatla2ee.com and dubizzle.com.eg for vehicle listings.
"""

import logging

from langchain_core.tools import tool
from opentelemetry import trace
from tavily import TavilyClient

from config import Config

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


@tool
def search_egyptian_cars(query: str) -> str:
    """Search for vehicles on Egyptian car marketplaces.

    Searches hatla2ee.com and dubizzle.com.eg for vehicle listings
    matching the provided query.

    Args:
        query: Constructed search query from extracted parameters
               (e.g., "Hyundai Tucson 2024 Egypt")

    Returns:
        Raw search results from Tavily API as text for LLM processing.
    """
    with tracer.start_as_current_span("tavily_vehicle_search") as span:
        config = Config()
        span.set_attribute("tool.input.query", query)

        try:
            client = TavilyClient(api_key=config.tavily_api_key)

            # Search with site filtering for Egyptian marketplaces
            results = client.search(
                query=query,
                search_depth="advanced",
                max_results=10,
                include_domains=["hatla2ee.com", "dubizzle.com.eg"],
            )

            result_count = len(results.get("results", []))
            span.set_attribute("tool.output.count", result_count)
            span.set_attribute("tool.output.status", "success")

            logger.info(f"Tavily search returned {result_count} results for: {query}")
            return str(results)

        except Exception as e:
            span.set_attribute("tool.output.status", "error")
            span.set_attribute("tool.output.error", str(e))
            logger.error(f"Tavily search error: {e}")
            return f"Error searching for vehicles: {str(e)}"
