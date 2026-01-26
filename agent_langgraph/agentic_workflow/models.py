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

from enum import Enum

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


class VehicleList(BaseModel):
    """List of extracted vehicles."""

    vehicles: list[Vehicle] = Field(
        ..., description="List of extracted vehicle details"
    )


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
        parts.append("for sale")
        parts.append("Egypt")  # Always include Egypt for location relevance
        return " ".join(parts)


class EmploymentType(str, Enum):
    """Employment types matching credit policies."""

    CORPORATE_MULTINATIONAL = "Corporate (Multinational)"
    CORPORATE_LOCAL = "Corporate (Local LLC)"
    SALARIED_PUBLIC = "Salaried (Public Sector)"
    SALARIED_PRIVATE = "Salaried (Private)"
    SELF_EMPLOYED_PROFESSIONAL = "Self-Employed (Professional)"
    SELF_EMPLOYED_COMMERCIAL = "Self-Employed (Commercial)"
    FREELANCER_TECH = "Freelancer (Tech)"
    FREELANCER_CREATIVE = "Freelancer (Creative)"
    RETIRED = "Retired / Pensioner"
    EXPAT = "Expat (Remittance)"


class UserProfile(BaseModel):
    """User profile for financing eligibility."""

    monthly_income: float | None = Field(None, description="Monthly income in EGP")
    employment_type: EmploymentType | None = Field(
        None, description="Employment category"
    )
    existing_debt_obligations: float = Field(
        0.0, description="Total existing monthly debt payments in EGP"
    )
    contact_name: str | None = Field(None, description="Full name")
    contact_phone: str | None = Field(None, description="Phone number")
    contact_email: str | None = Field(None, description="Email address")


class CreditPolicy(BaseModel):
    """Credit policy retrieved from RAG."""

    policy_id: str = Field(..., description="Unique policy identifier")
    employment_category: str = Field(..., description="Matching employment category")
    min_income: float = Field(..., description="Minimum income requirement")
    max_dbr: float = Field(..., description="Maximum Debt Burden Ratio (0.0 to 1.0)")
    description: str = Field(..., description="Policy description text")
    interest_rate: float = Field(15.0, description="Annual interest rate percentage")
    max_tenure_months: int = Field(60, description="Maximum loan tenure in months")


class LoanQuote(BaseModel):
    """Calculated loan quotation."""

    policy_id: str = Field(..., description="Reference policy ID")
    plan_name: str = Field(..., description="Plan display name")
    monthly_installment: float = Field(..., description="Calculated monthly payment")
    tenure_months: int = Field(..., description="Loan tenure in months")
    interest_rate: float = Field(..., description="Annual interest rate percentage")
    is_affordable: bool = Field(..., description="Does it pass DBR check?")
    dbr_percentage: float = Field(..., description="Calculated DBR for this loan")


class AgentState(MessagesState):
    """Extended state for Market Discovery and Financing agent.

    Extends MessagesState with additional fields for vehicle search and financing.
    """

    # Search Phase
    search_params: SearchParams | None = None
    search_results: list[Vehicle] = Field(default_factory=list)
    search_confirmed: bool = False

    # Selection
    selected_vehicle: Vehicle | None = None

    # Financing Phase
    user_profile: UserProfile | None = None
    eligible_policies: list[CreditPolicy] | None = None
    generated_quotes: list[LoanQuote] | None = None
    awaiting_submission: bool = (
        False  # Flag to indicate financing quotes were presented
    )

    # Internal
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
            "selected_vehicle": None,
            "user_profile": None,
            "eligible_policies": None,
            "generated_quotes": None,
            "awaiting_submission": False,
            "_next_node": None,
        }
