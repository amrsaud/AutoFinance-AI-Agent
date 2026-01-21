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
State schemas for the AutoFinance AI Agent.
Defines structured data models for vehicles, search parameters, customer info,
credit policies, and financial quotes.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ===== Structured Data Models =====


class Vehicle(BaseModel):
    """Structured schema for vehicle data from search results."""

    make: str = Field(description="Vehicle manufacturer (e.g., Toyota, Hyundai)")
    model: str = Field(description="Vehicle model name (e.g., Corolla, Tucson)")
    year: int = Field(description="Manufacturing year")
    price: float = Field(description="Listed price in EGP")
    mileage: int | None = Field(default=None, description="Odometer reading in km")
    source_url: str = Field(description="URL to the original listing")
    source_name: str = Field(description="Marketplace name (e.g., Hatla2ee, Dubizzle)")


class SearchParams(BaseModel):
    """Formalized search parameters extracted from user query."""

    make: str = Field(description="Vehicle make to search for")
    model: str = Field(description="Vehicle model to search for")
    year_from: int = Field(description="Minimum year (inclusive)")
    year_to: int = Field(description="Maximum year (inclusive)")
    price_cap: float | None = Field(default=None, description="Maximum price in EGP")


class CustomerInfo(BaseModel):
    """Customer personal information for lead capture."""

    full_name: str = Field(description="Customer's full name")
    email: str = Field(description="Contact email address")
    phone: str = Field(description="Phone number with country code")
    national_id: str | None = Field(
        default=None, description="Egyptian National ID (14 digits)"
    )


class CreditPolicy(BaseModel):
    """Credit policy retrieved from RAG."""

    interest_rate: float = Field(description="Annual interest rate as percentage")
    max_dbr: float = Field(description="Maximum Debt Burden Ratio allowed")
    min_income: float = Field(description="Minimum monthly income required in EGP")
    max_tenure_months: int = Field(description="Maximum loan tenure in months")
    max_vehicle_age: int = Field(description="Maximum vehicle age in years")
    eligible: bool = Field(
        description="Whether the user/vehicle combination is eligible"
    )
    rejection_reason: str | None = Field(
        default=None, description="Reason if not eligible"
    )


class FinancialQuote(BaseModel):
    """Calculated loan quotation."""

    principal: float = Field(description="Loan principal (vehicle price)")
    interest_rate: float = Field(description="Applied annual interest rate")
    tenure_months: int = Field(description="Loan duration in months")
    monthly_installment: float = Field(description="Monthly EMI amount")
    total_payment: float = Field(description="Total amount to be paid")
    total_interest: float = Field(description="Total interest over tenure")


# ===== Agent State =====


class AutoFinanceState(TypedDict):
    """Main state schema for the AutoFinance agent workflow."""

    messages: Annotated[list, add_messages]

    # Phase tracking
    current_phase: str

    # Observability (for DataRobot tracing)
    last_action: str | None
    error: str | None
    validation_result: str | None

    # Vehicle search
    search_params: SearchParams | None
    search_results: list[Vehicle] | None
    selected_vehicle: Vehicle | None

    # Financial profile
    monthly_income: float | None
    employment_type: str | None

    # Policy & calculation
    applicable_policy: CreditPolicy | None
    financial_quote: FinancialQuote | None
    risk_profile: dict | None  # DBR assessment

    # Lead capture
    customer_info: CustomerInfo | None
    request_id: str | None
