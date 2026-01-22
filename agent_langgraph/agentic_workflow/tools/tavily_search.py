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
Tavily Search API integration for Egyptian car marketplace search.

Searches across Hatla2ee and Dubizzle Egypt for vehicle listings
matching user criteria.
"""

import re
from typing import Optional

from config import Config
from langchain_core.tools import tool
from models import SearchParams, Vehicle
from tavily import TavilyClient

config = Config()


def _parse_price_from_text(text: str) -> Optional[float]:
    """
    Extract price from listing text.
    Handles formats like:
    - "1,500,000 EGP"
    - "EGP 1500000"
    - "1.5 million"
    """
    # Try to find price patterns
    patterns = [
        r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:EGP|جنيه|ج\.م)",  # 1,500,000 EGP
        r"(?:EGP|جنيه)\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)",  # EGP 1,500,000
        r"(\d+(?:\.\d+)?)\s*(?:million|مليون)",  # 1.5 million
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).replace(",", "")
            if "million" in text.lower() or "مليون" in text:
                return float(value) * 1_000_000
            return float(value)

    # Try simple number extraction as fallback
    numbers = re.findall(r"\d{4,}", text.replace(",", ""))
    if numbers:
        # Filter for reasonable car prices (50,000 - 10,000,000 EGP)
        for num in numbers:
            price = float(num)
            if 50_000 <= price <= 10_000_000:
                return price

    return None


def _parse_year_from_text(text: str) -> Optional[int]:
    """Extract model year from listing text."""
    # Look for 4-digit years between 2000-2030
    years = re.findall(r"\b(20[0-3]\d)\b", text)
    if years:
        return int(years[0])
    return None


def _parse_mileage_from_text(text: str) -> Optional[int]:
    """Extract mileage from listing text."""
    patterns = [
        r"(\d{1,3}(?:,\d{3})*)\s*(?:km|كم|kilometer)",  # 50,000 km
        r"(\d+)\s*(?:ألف|thousand)\s*(?:km|كم)",  # 50 ألف km
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).replace(",", "")
            if "ألف" in text or "thousand" in text.lower():
                return int(float(value) * 1000)
            return int(value)

    return None


def _build_search_query(params: SearchParams) -> str:
    """
    Build a search query optimized for Egyptian marketplaces.

    Template: "{Make} {Model} {Year} price in Egypt site:hatla2ee.com OR site:dubizzle.com.eg"
    """
    parts = []

    if params.make:
        parts.append(params.make)
    if params.model:
        parts.append(params.model)
    if params.year_min and params.year_max:
        if params.year_min == params.year_max:
            parts.append(str(params.year_min))
        else:
            parts.append(f"{params.year_min}-{params.year_max}")
    elif params.year_min:
        parts.append(f"{params.year_min}+")
    elif params.year_max:
        parts.append(str(params.year_max))

    parts.append("price in Egypt")

    query = " ".join(parts)
    # Add domain constraints for Egyptian marketplaces
    query += " site:hatla2ee.com OR site:dubizzle.com.eg"

    return query


@tool
def search_vehicles(
    make: str,
    model: str,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    price_cap: Optional[float] = None,
    max_results: int = 5,
) -> list[Vehicle]:
    """
    Search for vehicles on Egyptian marketplaces (Hatla2ee, Dubizzle).

    Args:
        make: Vehicle manufacturer (e.g., "Hyundai", "Toyota")
        model: Vehicle model (e.g., "Tucson", "Corolla")
        year_min: Minimum model year (optional)
        year_max: Maximum model year (optional)
        price_cap: Maximum price in EGP (optional)
        max_results: Maximum number of results to return

    Returns:
        List of Vehicle objects with price, year, mileage, and source URL
    """
    if not config.tavily_api_key:
        raise ValueError("TAVILY_API_KEY not configured")

    client = TavilyClient(api_key=config.tavily_api_key)

    params = SearchParams(
        make=make,
        model=model,
        year_min=year_min,
        year_max=year_max,
        price_cap=price_cap,
    )

    query = _build_search_query(params)

    # Execute Tavily search
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results * 2,  # Get more to filter
        include_domains=["hatla2ee.com", "dubizzle.com.eg"],
    )

    vehicles = []

    for result in response.get("results", []):
        title = result.get("title", "")
        content = result.get("content", "")
        url = result.get("url", "")
        combined_text = f"{title} {content}"

        # Parse vehicle details
        price = _parse_price_from_text(combined_text)
        year = _parse_year_from_text(combined_text)
        mileage = _parse_mileage_from_text(combined_text)

        # Skip if no price found
        if price is None:
            continue

        # Apply price cap filter
        if price_cap and price > price_cap:
            continue

        # Apply year filter
        if year:
            if year_min and year < year_min:
                continue
            if year_max and year > year_max:
                continue

        # Determine source site
        source_site = "Hatla2ee" if "hatla2ee" in url.lower() else "Dubizzle"

        vehicle = Vehicle(
            name=title or f"{make} {model}",
            price=price,
            year=year or (year_min or 2020),
            mileage=mileage,
            source_url=url,
            source_site=source_site,
        )
        vehicles.append(vehicle)

        if len(vehicles) >= max_results:
            break

    return vehicles


def search_vehicles_sync(params: SearchParams, max_results: int = 5) -> list[Vehicle]:
    """
    Synchronous wrapper for vehicle search.

    This is used by the LangGraph nodes directly.
    """
    return search_vehicles.invoke(
        {
            "make": params.make or "",
            "model": params.model or "",
            "year_min": params.year_min,
            "year_max": params.year_max,
            "price_cap": params.price_cap,
            "max_results": max_results,
        }
    )
