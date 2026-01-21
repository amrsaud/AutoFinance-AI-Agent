# Copyright 2025 DataRobot, Inc.
# Licensed under the Apache License, Version 2.0
"""Supabase client for lead storage."""

from datetime import datetime

from langchain_core.tools import tool
from supabase import Client, create_client

try:
    from config import Config
    from state import CustomerInfo, FinancialQuote, Vehicle
except ImportError:
    from ..config import Config
    from ..state import CustomerInfo, FinancialQuote, Vehicle


def _get_supabase_client() -> Client | None:
    """Create Supabase client from config."""
    config = Config()
    if not config.supabase_url or not config.supabase_key:
        return None
    return create_client(config.supabase_url, config.supabase_key)


@tool
def save_application(
    customer: CustomerInfo,
    vehicle: Vehicle,
    quote: FinancialQuote,
    monthly_income: float,
    employment_type: str,
) -> str:
    """Save loan application to Supabase.

    Returns request_id (e.g., 'AF-260119-1234')
    """
    client = _get_supabase_client()

    if not client:
        timestamp = datetime.now().strftime("%y%m%d%H%M%S")
        return f"AF-{timestamp}-MOCK"

    try:
        data = {
            "customer_name": customer.full_name,
            "customer_email": customer.email,
            "customer_phone": customer.phone,
            "national_id": customer.national_id,
            "vehicle_name": f"{vehicle.year} {vehicle.make} {vehicle.model}",
            "vehicle_price": vehicle.price,
            "vehicle_year": vehicle.year,
            "vehicle_mileage": vehicle.mileage,
            "vehicle_source_url": vehicle.source_url,
            "monthly_income": monthly_income,
            "employment_type": employment_type,
            "interest_rate": quote.interest_rate,
            "tenure_months": quote.tenure_months,
            "monthly_installment": quote.monthly_installment,
            "total_payment": quote.total_payment,
            "status": "pending_review",
        }

        response = client.table("applications").insert(data).execute()
        if response.data:
            return response.data[0].get("request_id", "ERROR-NO-ID")
        return "ERROR-INSERT-FAILED"
    except Exception as e:
        return f"ERROR-{str(e)[:20]}"


@tool
def check_application_status(request_id: str) -> dict:
    """Check status of existing application.

    Args:
        request_id: Request ID (e.g., 'AF-260119-1234')
    """
    client = _get_supabase_client()

    if not client:
        return {"request_id": request_id, "status": "unknown", "found": False}

    try:
        response = (
            client.table("applications")
            .select("request_id, status, vehicle_name, created_at")
            .eq("request_id", request_id)
            .execute()
        )

        if response.data:
            record = response.data[0]
            return {
                "request_id": record["request_id"],
                "status": record["status"],
                "vehicle_name": record["vehicle_name"],
                "created_at": record["created_at"],
                "found": True,
            }
        return {"request_id": request_id, "status": "not_found", "found": False}
    except Exception:
        return {"request_id": request_id, "status": "error", "found": False}
