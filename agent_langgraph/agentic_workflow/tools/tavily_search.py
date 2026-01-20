# Copyright 2025 DataRobot, Inc.
# Licensed under the Apache License, Version 2.0
"""Tavily Search tool for Egyptian car marketplaces."""

from langchain_core.tools import tool
from tavily import TavilyClient

from ..config import Config
from ..state import SearchParams


@tool
def search_vehicles(params: dict) -> list[dict]:
    """Search Egyptian marketplaces for vehicles.

    Args:
        params: Dict with make, model, year_from, year_to, price_cap
    """
    # Convert dict to SearchParams if needed (just for field access, or use dict directly)
    if isinstance(params, dict):
        search_params = SearchParams(**params)
    else:
        search_params = params

    config = Config()
    if not config.tavily_api_key:
        return []

    client = TavilyClient(api_key=config.tavily_api_key)

    query = f"{search_params.make} {search_params.model} {search_params.year_from}-{search_params.year_to} سيارة للبيع مصر"
    if search_params.price_cap:
        query += f" تحت {int(search_params.price_cap)} جنيه"

    include_domains = ["hatla2ee.com", "dubizzle.com.eg", "olx.com.eg"]

    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            include_domains=include_domains,
            max_results=8,
        )
        return response.get("results", [])

    except Exception as e:
        print(f"Tavily search error: {e}")
        return []
