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
Supabase integration for application storage and status queries.

Provides functionality to:
- Store loan applications for back-office review
- Query application status by Request ID
"""

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from config import Config
from models import (
    ApplicationStatus,
    CustomerInfo,
    FinancialQuote,
    Vehicle,
)
from supabase import Client, create_client

config = Config()


def _get_supabase_client() -> Client:
    """Get configured Supabase client."""
    if not config.supabase_url or not config.supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be configured")
    return create_client(config.supabase_url, config.supabase_key)


def create_application(
    session_id: str,
    customer_info: CustomerInfo,
    vehicle: Vehicle,
    financial_quote: FinancialQuote,
    monthly_income: float,
    employment_type: str,
) -> str:
    """
    Create a new loan application in Supabase.

    Args:
        session_id: The conversation thread ID
        customer_info: Customer personal information
        vehicle: Selected vehicle details
        financial_quote: Calculated loan details
        monthly_income: User's monthly income
        employment_type: User's employment type

    Returns:
        The generated request_id (UUID)
    """
    client = _get_supabase_client()

    request_id = str(uuid4())

    application_data = {
        "request_id": request_id,
        "session_id": session_id,
        "user_name": customer_info.full_name,
        "contact_details": {
            "email": customer_info.email,
            "phone": customer_info.phone,
            "national_id": customer_info.national_id,
        },
        "vehicle_summary": {
            "name": vehicle.name,
            "price": vehicle.price,
            "year": vehicle.year,
            "mileage": vehicle.mileage,
            "source_url": vehicle.source_url,
        },
        "financial_summary": {
            "monthly_income": monthly_income,
            "employment_type": employment_type,
            "monthly_installment": financial_quote.monthly_installment,
            "interest_rate": financial_quote.interest_rate,
            "tenure_months": financial_quote.tenure_months,
            "total_amount": financial_quote.total_amount,
        },
        "status": ApplicationStatus.PENDING_REVIEW.value,
        "created_at": datetime.utcnow().isoformat(),
    }

    client.table("applications").insert(application_data).execute()

    return request_id


def get_application_status(request_id: str) -> Optional[dict[str, Any]]:
    """
    Query application status by Request ID.

    Args:
        request_id: The unique request ID

    Returns:
        Application data dict if found, None otherwise
    """
    client = _get_supabase_client()

    result = (
        client.table("applications")
        .select(
            "request_id, user_name, status, created_at, vehicle_summary, financial_summary"
        )
        .eq("request_id", request_id)
        .execute()
    )

    if result.data and len(result.data) > 0:
        return result.data[0]

    return None


def update_application_status(request_id: str, new_status: ApplicationStatus) -> bool:
    """
    Update the status of an existing application.

    Args:
        request_id: The unique request ID
        new_status: The new status to set

    Returns:
        True if update successful, False otherwise
    """
    client = _get_supabase_client()

    result = (
        client.table("applications")
        .update({"status": new_status.value})
        .eq("request_id", request_id)
        .execute()
    )

    return len(result.data) > 0


def format_status_response(application: dict[str, Any]) -> str:
    """
    Format application status for display to user.

    Args:
        application: Application data from database

    Returns:
        Formatted status message
    """
    status_messages = {
        "pending_review": "⏳ **Pending Review** - Your application is awaiting review by our team.",
        "under_review": "🔍 **Under Review** - Our team is currently reviewing your application.",
        "approved": "✅ **Approved** - Congratulations! Your loan has been approved.",
        "rejected": "❌ **Rejected** - Unfortunately, your application was not approved.",
        "documents_required": "📄 **Documents Required** - Please submit additional documents.",
    }

    vehicle = application.get("vehicle_summary", {})
    financial = application.get("financial_summary", {})
    status = application.get("status", "pending_review")

    response = f"""
## Application Status: {application.get("request_id")}

**Applicant:** {application.get("user_name")}
**Submitted:** {application.get("created_at", "Unknown")}

### Vehicle
- **{vehicle.get("name", "N/A")}**
- Price: {vehicle.get("price", 0):,.0f} EGP

### Loan Details
- Monthly Installment: {financial.get("monthly_installment", 0):,.0f} EGP
- Tenure: {financial.get("tenure_months", 60)} months

### Status
{status_messages.get(status, "Status unknown")}
"""
    return response.strip()
