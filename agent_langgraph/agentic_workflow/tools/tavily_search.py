# Copyright 2025 DataRobot, Inc.
# Licensed under the Apache License, Version 2.0
"""Tavily Search tool for Egyptian car marketplaces."""

import re
from datetime import datetime

from langchain_core.tools import tool
from tavily import TavilyClient

try:
    from config import Config
    from state import SearchParams, Vehicle
except ImportError:
    from ..config import Config
    from ..state import SearchParams, Vehicle


def _parse_vehicle_from_result(result: dict, params: SearchParams) -> Vehicle | None:
    """Parse a Tavily search result into a Vehicle object."""
    url = result.get("url", "")
    title = result.get("title", "")
    content = result.get("content", "")

    # Determine source
    source_name = "Unknown"
    if "hatla2ee" in url.lower():
        source_name = "Hatla2ee"
    elif "dubizzle" in url.lower():
        source_name = "Dubizzle Egypt"
    elif "olx" in url.lower():
        source_name = "OLX Egypt"

    # Extract price
    price_match = re.search(r"([\d,]+)\s*(?:EGP|جنيه)", content)
    if not price_match:
        price_match = re.search(r"(\d{3,}(?:,\d{3})*)", content)

    price = 0.0
    if price_match:
        try:
            price = float(price_match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Extract mileage
    mileage = None
    mileage_match = re.search(r"(\d+(?:,\d+)?)\s*(?:km|كم)", content, re.IGNORECASE)
    if mileage_match:
        try:
            mileage = int(mileage_match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Extract year
    year_match = re.search(r"\b(20\d{2}|19\d{2})\b", title + " " + content)
    year = params.year_from
    if year_match:
        try:
            year = int(year_match.group(1))
        except ValueError:
            pass

    if price <= 0:
        return None

    return Vehicle(
        make=params.make,
        model=params.model,
        year=year,
        price=price,
        mileage=mileage,
        source_url=url,
        source_name=source_name,
    )


@tool
def search_vehicles(params: SearchParams) -> list[Vehicle]:
    """Search Egyptian marketplaces for vehicles.

    Args:
        params: SearchParams with make, model, year range, price cap
    """
    config = Config()

    if not config.tavily_api_key:
        return []

    client = TavilyClient(api_key=config.tavily_api_key)
    current_year = datetime.now().year

    query = f"{params.make} {params.model} {params.year_from}-{params.year_to} سيارة للبيع مصر"
    if params.price_cap:
        query += f" تحت {int(params.price_cap)} جنيه"

    include_domains = ["hatla2ee.com", "dubizzle.com.eg", "olx.com.eg"]

    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            include_domains=include_domains,
            max_results=10,
        )

        vehicles: list[Vehicle] = []
        for result in response.get("results", []):
            vehicle = _parse_vehicle_from_result(result, params)
            if vehicle:
                if params.price_cap and vehicle.price > params.price_cap:
                    continue
                if vehicle.year < params.year_from or vehicle.year > params.year_to:
                    continue
                if current_year - vehicle.year > config.max_vehicle_age:
                    continue
                vehicles.append(vehicle)

        return vehicles
    except Exception as e:
        print(f"Tavily search error: {e}")
        return []
