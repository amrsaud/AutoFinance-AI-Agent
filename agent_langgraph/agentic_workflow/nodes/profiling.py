# Copyright 2025 DataRobot, Inc.
# Licensed under the Apache License, Version 2.0
"""Profiling node - collects income, checks risk profile, calculates quote."""

import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

try:
    from state import AutoFinanceState
    from tools.policy_rag import get_credit_policy
    from tools.calculator import calculate_installment, check_debt_burden_ratio
except ImportError:
    from ..state import AutoFinanceState
    from ..tools.policy_rag import get_credit_policy
    from ..tools.calculator import calculate_installment, check_debt_burden_ratio


async def profiling_node(
    state: AutoFinanceState, config: RunnableConfig = None
) -> dict[str, Any]:
    """Collect financial info, check risk profile, and calculate quote.

    This node:
    1. Extracts income and employment type from user message
    2. Checks credit policy eligibility
    3. Calculates Debt Burden Ratio (risk profile)
    4. Generates loan quote if eligible
    """
    vehicle = state.get("selected_vehicle")
    messages = state.get("messages", [])

    if not vehicle:
        return {
            "messages": [AIMessage(content="Please select a vehicle first.")],
            "current_phase": "onboarding",
        }

    # Get last user message
    last_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_msg = msg.content.lower()
            break

    # Extract income
    income = None
    income_match = re.search(r"(\d+[,\d]*)", last_msg.replace(",", ""))
    if income_match:
        try:
            income = float(income_match.group(1))
        except ValueError:
            pass

    # Extract employment type
    employment = None
    if any(w in last_msg for w in ["salaried", "salary", "موظف"]):
        employment = "salaried"
    elif any(w in last_msg for w in ["self", "business", "حر", "اعمال"]):
        employment = "self_employed"
    elif any(w in last_msg for w in ["corporate", "company", "شركة"]):
        employment = "corporate"

    # Check if we need more info
    if not income or not employment:
        return {
            "messages": [
                AIMessage(
                    content="Please provide your monthly income and employment type.\nExample: 'My income is 25,000 EGP and I'm salaried'"
                )
            ],
            "monthly_income": income,
            "employment_type": employment,
            "current_phase": "profiling",
        }

    # Handle vehicle as Pydantic or dict
    if hasattr(vehicle, "model_dump"):
        vehicle_data = vehicle
    elif isinstance(vehicle, dict):
        from ..state import Vehicle

        vehicle_data = Vehicle(**vehicle)
    else:
        vehicle_data = vehicle

    # Check credit policy
    policy = get_credit_policy.invoke(
        {
            "vehicle": vehicle_data,
            "monthly_income": income,
            "employment_type": employment,
        },
        config=config,
    )

    if not policy.eligible:
        return {
            "messages": [
                AIMessage(
                    content=f"❌ **Unable to proceed**\n\n{policy.rejection_reason}\n\nWould you like to search for a different vehicle?"
                )
            ],
            "monthly_income": income,
            "employment_type": employment,
            "applicable_policy": policy,
            "current_phase": "onboarding",
        }

    # Calculate quote
    vehicle_price = (
        vehicle_data.price
        if hasattr(vehicle_data, "price")
        else vehicle_data.get("price", 0)
    )
    quote = calculate_installment.invoke(
        {
            "principal": vehicle_price,
            "annual_rate": policy.interest_rate,
            "tenure_months": policy.max_tenure_months,
        },
        config=config,
    )

    # Check Debt Burden Ratio (Risk Profile)
    dbr_result = check_debt_burden_ratio.invoke(
        {
            "monthly_income": income,
            "monthly_installment": quote.monthly_installment,
            "max_dbr": policy.max_dbr,
        },
        config=config,
    )

    if not dbr_result.get("eligible"):
        return {
            "messages": [
                AIMessage(
                    content=f"""❌ **High Risk Profile**

{dbr_result.get("reason")}

Your monthly installment ({quote.monthly_installment:,.0f} EGP) is too high relative to your income ({income:,.0f} EGP).

**Options:**
- Look for a less expensive vehicle
- Consider a longer tenure to reduce monthly payments

Would you like to search for a different vehicle?"""
                )
            ],
            "monthly_income": income,
            "employment_type": employment,
            "applicable_policy": policy,
            "financial_quote": quote,
            "risk_profile": dbr_result,
            "current_phase": "onboarding",
        }

    # Build vehicle info string
    if hasattr(vehicle_data, "year"):
        vehicle_str = f"{vehicle_data.year} {vehicle_data.make} {vehicle_data.model}"
    else:
        vehicle_str = f"{vehicle_data.get('year')} {vehicle_data.get('make')} {vehicle_data.get('model')}"

    return {
        "messages": [
            AIMessage(
                content=f"""💳 **Your Loan Quote**

🚗 **{vehicle_str}** - {vehicle_price:,.0f} EGP

📊 **Terms:**
- Interest Rate: {quote.interest_rate:.1f}% per annum
- Tenure: {quote.tenure_months} months
- **Monthly Installment: {quote.monthly_installment:,.0f} EGP**
- Total Payment: {quote.total_payment:,.0f} EGP

✅ **Risk Assessment:** DBR {dbr_result["dbr"]:.1%} (Max: {dbr_result["max_allowed"]:.1%})

**Ready to apply?** (Yes to proceed)"""
            )
        ],
        "monthly_income": income,
        "employment_type": employment,
        "applicable_policy": policy,
        "financial_quote": quote,
        "risk_profile": dbr_result,
        "current_phase": "quotation",
    }
