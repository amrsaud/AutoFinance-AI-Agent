# Copyright 2025 DataRobot, Inc.
# Licensed under the Apache License, Version 2.0
"""Loan calculation utilities."""

from langchain_core.tools import tool

try:
    from state import FinancialQuote
except ImportError:
    from ..state import FinancialQuote


@tool
def calculate_installment(
    principal: float,
    annual_rate: float,
    tenure_months: int,
) -> FinancialQuote:
    """Calculate monthly installment (EMI) using amortization formula.

    Args:
        principal: Loan principal (vehicle price) in EGP
        annual_rate: Annual interest rate as percentage (e.g., 18.0)
        tenure_months: Loan duration in months
    """
    monthly_rate = (annual_rate / 100) / 12

    if monthly_rate == 0:
        monthly_installment = principal / tenure_months
        total_payment = principal
        total_interest = 0.0
    else:
        rate_factor = (1 + monthly_rate) ** tenure_months
        monthly_installment = principal * monthly_rate * rate_factor / (rate_factor - 1)
        total_payment = monthly_installment * tenure_months
        total_interest = total_payment - principal

    return FinancialQuote(
        principal=round(principal, 2),
        interest_rate=annual_rate,
        tenure_months=tenure_months,
        monthly_installment=round(monthly_installment, 2),
        total_payment=round(total_payment, 2),
        total_interest=round(total_interest, 2),
    )


@tool
def check_debt_burden_ratio(
    monthly_income: float,
    monthly_installment: float,
    max_dbr: float,
) -> dict:
    """Check if installment fits within allowed Debt Burden Ratio.

    Args:
        monthly_income: Customer's monthly income in EGP
        monthly_installment: Calculated monthly EMI in EGP
        max_dbr: Maximum allowed DBR as decimal (e.g., 0.40)
    """
    if monthly_income <= 0:
        return {
            "dbr": 1.0,
            "max_allowed": max_dbr,
            "eligible": False,
            "reason": "Invalid income",
        }

    dbr = monthly_installment / monthly_income
    return {
        "dbr": round(dbr, 4),
        "max_allowed": max_dbr,
        "eligible": dbr <= max_dbr,
        "reason": None if dbr <= max_dbr else f"DBR {dbr:.1%} exceeds {max_dbr:.1%}",
    }
