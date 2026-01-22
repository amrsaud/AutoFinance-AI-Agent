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
Pydantic models for the AutoFinance AI Agent state and data structures.
These models define the runtime agent state and ensure type safety
throughout the workflow.
"""

from enum import Enum
from typing import Annotated, Optional
from uuid import uuid4

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class WorkflowPhase(str, Enum):
    """Tracks the macro-status of the conversation."""

    ONBOARDING = "onboarding"
    DISCOVERY = "discovery"
    PROFILING = "profiling"
    QUOTATION = "quotation"
    SUBMISSION = "submission"
    COMPLETED = "completed"


class EmploymentType(str, Enum):
    """User employment categories for credit assessment."""

    SALARIED = "salaried"
    SELF_EMPLOYED = "self_employed"
    CORPORATE = "corporate"
    OTHER = "other"


class SearchParams(BaseModel):
    """Structured search query extracted from user input."""

    make: Optional[str] = Field(
        None, description="Vehicle manufacturer (e.g., Hyundai, Toyota)"
    )
    model: Optional[str] = Field(
        None, description="Vehicle model (e.g., Tucson, Corolla)"
    )
    year_min: Optional[int] = Field(None, description="Minimum year for search")
    year_max: Optional[int] = Field(None, description="Maximum year for search")
    price_cap: Optional[float] = Field(None, description="Maximum price in EGP")


class Vehicle(BaseModel):
    """Vehicle listing from marketplace search results."""

    name: str = Field(..., description="Full vehicle name/title")
    price: float = Field(..., description="Listed price in EGP")
    year: int = Field(..., description="Model year")
    mileage: Optional[int] = Field(None, description="Mileage in kilometers")
    source_url: str = Field(..., description="Link to the original listing")
    source_site: Optional[str] = Field(
        None, description="Source marketplace (e.g., Hatla2ee, Dubizzle)"
    )


class CreditPolicy(BaseModel):
    """Credit policy retrieved via RAG from the vector database."""

    interest_rate: float = Field(
        ..., description="Annual interest rate as decimal (e.g., 0.18 for 18%)"
    )
    max_tenure_months: int = Field(60, description="Maximum loan tenure in months")
    max_debt_burden_ratio: float = Field(0.5, description="Maximum DBR allowed")
    min_income: float = Field(
        5000.0, description="Minimum monthly income required in EGP"
    )
    max_vehicle_age: int = Field(10, description="Maximum vehicle age in years")
    is_eligible: bool = Field(
        True, description="Whether the user meets eligibility criteria"
    )
    rejection_reason: Optional[str] = Field(
        None, description="Reason for rejection if not eligible"
    )


class FinancialQuote(BaseModel):
    """Calculated loan details based on PMT formula."""

    principal: float = Field(
        ..., description="Loan principal amount (car price) in EGP"
    )
    interest_rate: float = Field(..., description="Applied annual interest rate")
    tenure_months: int = Field(..., description="Loan tenure in months")
    monthly_installment: float = Field(..., description="Monthly payment in EGP")
    total_interest: float = Field(
        ..., description="Total interest over loan lifetime in EGP"
    )
    total_amount: float = Field(
        ..., description="Total amount payable (principal + interest)"
    )


class CustomerInfo(BaseModel):
    """Personal information collected at submission."""

    full_name: str = Field(..., description="Customer's full legal name")
    email: str = Field(..., description="Contact email address")
    phone: str = Field(..., description="Contact phone number")
    national_id: Optional[str] = Field(
        None, description="Egyptian National ID (optional)"
    )


class ApplicationStatus(str, Enum):
    """Status of a loan application."""

    PENDING_REVIEW = "pending_review"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DOCUMENTS_REQUIRED = "documents_required"


class AgentState(BaseModel):
    """
    Complete runtime state for the AutoFinance AI Agent.
    This schema defines the 'Short-Term Memory' of the agent.
    """

    # Conversation history - using LangGraph's add_messages reducer
    messages: Annotated[list[AnyMessage], add_messages] = Field(
        default_factory=list, description="Append-only log of all Human and AI messages"
    )

    # Workflow tracking
    current_phase: WorkflowPhase = Field(
        default=WorkflowPhase.ONBOARDING,
        description="Current macro-status of the workflow",
    )

    # Phase 2: Market Discovery
    search_params: Optional[SearchParams] = Field(
        None, description="Structured query extracted from user input"
    )
    search_params_confirmed: bool = Field(
        False, description="Whether user has confirmed the search parameters"
    )
    search_results: list[Vehicle] = Field(
        default_factory=list, description="List of vehicles returned by Tavily search"
    )
    selected_vehicle: Optional[Vehicle] = Field(
        None, description="The vehicle the user has chosen to finance"
    )

    # Phase 3: Financial Profiling
    monthly_income: Optional[float] = Field(
        None, description="User's self-reported monthly income in EGP"
    )
    employment_type: Optional[EmploymentType] = Field(
        None, description="User's employment category"
    )
    applicable_policy: Optional[CreditPolicy] = Field(
        None, description="Credit policy retrieved via RAG"
    )

    # Phase 4: Quotation
    financial_quote: Optional[FinancialQuote] = Field(
        None, description="Calculated loan details"
    )
    quote_confirmed: bool = Field(
        False, description="Whether user has confirmed the financial quote"
    )

    # Phase 5: Submission
    customer_info: Optional[CustomerInfo] = Field(
        None, description="PII collected at submission"
    )
    request_id: Optional[str] = Field(
        None, description="Unique UUID generated upon successful submission"
    )

    # Utility fields
    awaiting_confirmation: Optional[str] = Field(
        None,
        description="Type of confirmation being awaited (e.g., 'search', 'quote', 'submit')",
    )
    error_message: Optional[str] = Field(
        None, description="Error message to display to user if something went wrong"
    )

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def generate_request_id(cls) -> str:
        """Generate a unique request ID for submissions."""
        return str(uuid4())
