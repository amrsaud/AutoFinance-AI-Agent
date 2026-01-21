# Copyright 2025 DataRobot, Inc.
# Licensed under the Apache License, Version 2.0
"""Quotation node - uses interrupt() for human confirmation of loan terms."""

from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt

try:
    from state import AutoFinanceState
except ImportError:
    from ..state import AutoFinanceState


async def quotation_node(
    state: AutoFinanceState, config: RunnableConfig = None
) -> Command[Literal["lead_capture"]] | dict[str, Any]:
    """Display quote and wait for user confirmation using interrupt().

    This node uses LangGraph's interrupt pattern to get user acceptance/rejection.
    """
    quote = state.get("financial_quote")
    vehicle = state.get("selected_vehicle")
    risk_profile = state.get("risk_profile", {})

    if not quote or not vehicle:
        return {
            "messages": [AIMessage(content="Something went wrong. Please start over.")],
            "current_phase": "onboarding",
        }

    # Build quote summary for interrupt payload
    if hasattr(vehicle, "model_dump"):
        vd = vehicle.model_dump()
    elif isinstance(vehicle, dict):
        vd = vehicle
    else:
        vd = {
            "year": vehicle.year,
            "make": vehicle.make,
            "model": vehicle.model,
            "price": vehicle.price,
        }

    if hasattr(quote, "model_dump"):
        qd = quote.model_dump()
    elif isinstance(quote, dict):
        qd = quote
    else:
        qd = {
            "interest_rate": quote.interest_rate,
            "tenure_months": quote.tenure_months,
            "monthly_installment": quote.monthly_installment,
            "total_payment": quote.total_payment,
        }

    # Pause for user confirmation
    decision = interrupt(
        {
            "type": "quotation_confirmation",
            "vehicle": f"{vd.get('year')} {vd.get('make')} {vd.get('model')}",
            "vehicle_price": vd.get("price", 0),
            "interest_rate": qd.get("interest_rate"),
            "tenure_months": qd.get("tenure_months"),
            "monthly_installment": qd.get("monthly_installment"),
            "total_payment": qd.get("total_payment"),
            "dbr": risk_profile.get("dbr"),
            "prompt": "Do you want to proceed with this loan application? (yes/no)",
        }
    )

    # Resume with user's decision
    decision_str = str(decision).lower()

    if any(
        w in decision_str for w in ["yes", "proceed", "apply", "submit", "نعم", "موافق"]
    ):
        return Command(
            goto="lead_capture",
            update={
                "current_phase": "lead_capture",
                "messages": [
                    AIMessage(
                        content="""📝 **Almost done!**

Please provide your contact information:
- **Full Name**
- **Email**
- **Phone** (e.g., +201234567890)

Example: "Ahmed Mohamed, ahmed@email.com, +201234567890" """
                    )
                ],
            },
        )

    # User declined
    return {
        "messages": [
            AIMessage(
                content="No problem! Would you like to search for a different vehicle or adjust the terms?"
            )
        ],
        "current_phase": "onboarding",
    }
