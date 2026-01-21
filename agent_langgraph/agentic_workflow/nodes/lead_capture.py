# Copyright 2025 DataRobot, Inc.
# Licensed under the Apache License, Version 2.0
"""Lead capture node - collects contact info and saves application."""

import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

try:
    from state import AutoFinanceState, CustomerInfo
    from tools.supabase_client import save_application
except ImportError:
    from ..state import AutoFinanceState, CustomerInfo
    from ..tools.supabase_client import save_application


async def lead_capture_node(
    state: AutoFinanceState, config: RunnableConfig = None
) -> dict[str, Any]:
    """Collect contact info and save application to database."""
    vehicle = state.get("selected_vehicle")
    quote = state.get("financial_quote")
    messages = state.get("messages", [])

    if not vehicle or not quote:
        return {
            "messages": [AIMessage(content="Something went wrong. Please start over.")],
            "current_phase": "onboarding",
        }

    # Get last user message
    last_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_msg = msg.content
            break

    # Parse contact info
    email_match = re.search(r"[\w.-]+@[\w.-]+\.\w+", last_msg)
    phone_match = re.search(r"\+?\d[\d\s-]{8,}", last_msg)

    # Try to extract name (first part before comma or email)
    parts = last_msg.split(",")
    name = parts[0].strip() if parts else "Customer"

    # Clean name if it contains email
    if "@" in name:
        name = "Customer"

    if not email_match:
        return {
            "messages": [
                AIMessage(
                    content="Please provide a valid email address.\nFormat: Name, email@example.com, +20XXXXXXXXXX"
                )
            ],
            "current_phase": "lead_capture",
        }

    customer = CustomerInfo(
        full_name=name,
        email=email_match.group(0),
        phone=phone_match.group(0).strip() if phone_match else "Not provided",
    )

    # Handle vehicle as Pydantic or dict
    if hasattr(vehicle, "model_dump"):
        vehicle_data = vehicle
    elif isinstance(vehicle, dict):
        from ..state import Vehicle

        vehicle_data = Vehicle(**vehicle)
    else:
        vehicle_data = vehicle

    # Handle quote as Pydantic or dict
    if hasattr(quote, "model_dump"):
        quote_data = quote
    elif isinstance(quote, dict):
        from ..state import FinancialQuote

        quote_data = FinancialQuote(**quote)
    else:
        quote_data = quote

    # Save application
    request_id = save_application.invoke(
        {
            "customer": customer,
            "vehicle": vehicle_data,
            "quote": quote_data,
            "monthly_income": state.get("monthly_income", 0),
            "employment_type": state.get("employment_type", ""),
        },
        config=config,
    )

    # Build vehicle string
    if hasattr(vehicle_data, "year"):
        vehicle_str = f"{vehicle_data.year} {vehicle_data.make} {vehicle_data.model}"
        monthly = quote_data.monthly_installment
    else:
        vehicle_str = f"{vehicle_data.get('year')} {vehicle_data.get('make')} {vehicle_data.get('model')}"
        monthly = quote_data.get("monthly_installment", 0)

    return {
        "messages": [
            AIMessage(
                content=f"""🎉 **Application Submitted!**

**Your Request ID: {request_id}**

📋 **Summary:**
- Vehicle: {vehicle_str}
- Monthly: {monthly:,.0f} EGP

We'll contact you at {customer.email} within 24-48 hours.

Save your Request ID: **{request_id}**

Thank you for choosing AutoFinance! 🚗"""
            )
        ],
        "customer_info": customer,
        "request_id": request_id,
        "current_phase": "onboarding",
    }
