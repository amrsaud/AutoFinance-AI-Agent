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
Pydantic models for the AutoFinance AI Agent state management.
Defines Vehicle, SearchParams, and AgentState for market discovery.
"""

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class Vehicle(BaseModel):
    """Vehicle listing from Egyptian car marketplaces."""

    make: str = Field(..., description="Car manufacturer, e.g., 'Hyundai'")
    model: str = Field(..., description="Car model, e.g., 'Tucson'")
    year: int | None = Field(None, description="Manufacturing year, e.g., 2024")
    price: int | None = Field(None, description="Price in EGP")
    mileage: int | None = Field(None, description="Mileage in kilometers")
    location: str | None = Field(
        None, description="Location, e.g., 'Cairo', 'Alexandria'"
    )
    source: str = Field(..., description="Site name: 'hatla2ee' or 'dubizzle'")
    source_url: str = Field(..., description="Full URL to the vehicle listing")


class SearchParams(BaseModel):
    """Search parameters extracted from natural language user queries."""

    make: str | None = Field(None, description="Car manufacturer")
    model: str | None = Field(None, description="Car model")
    year_min: int | None = Field(None, description="Minimum year filter")
    year_max: int | None = Field(None, description="Maximum year filter")
    price_max: int | None = Field(None, description="Maximum price in EGP")
    raw_query: str = Field(default="", description="Original user query")

    def build_search_query(self) -> str:
        """Construct consistent search query from extracted parameters.

        Returns:
            str: A search query string built from the extracted parameters.
        """
        parts = []
        if self.make:
            parts.append(self.make)
        if self.model:
            parts.append(self.model)
        if self.year_min:
            parts.append(str(self.year_min))
        parts.append("Egypt")  # Always include Egypt for location relevance
        return " ".join(parts)


class AgentState(MessagesState):
    """Extended state for Market Discovery agent.

    Extends MessagesState with additional fields for vehicle search tracking.
    """

    search_params: SearchParams | None = None
    search_results: list[Vehicle] = Field(default_factory=list)
    search_confirmed: bool = False
    _next_node: str | None = None  # Router's routing decision

    @classmethod
    def get_initial_state(cls) -> dict:
        """Return a fresh initial state for reset.

        Returns:
            dict: Initial state with empty values.
        """
        return {
            "messages": [],
            "search_params": None,
            "search_results": [],
            "search_confirmed": False,
            "_next_node": None,
        }
