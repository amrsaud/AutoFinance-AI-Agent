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
PMT-based installment calculator for loan quotations.

Uses the standard Present Value of Annuity formula (PMT) to calculate
monthly installments.
"""

from models import FinancialQuote


def calculate_monthly_installment(
    principal: float,
    annual_interest_rate: float,
    tenure_months: int,
) -> FinancialQuote:
    """
    Calculate monthly installment using the PMT formula.

    PMT = P * [r(1+r)^n] / [(1+r)^n - 1]

    Where:
    - P = Principal (loan amount)
    - r = Monthly interest rate (annual rate / 12)
    - n = Number of months

    Args:
        principal: Loan amount (car price) in EGP
        annual_interest_rate: Annual interest rate as decimal (e.g., 0.18 for 18%)
        tenure_months: Loan tenure in months

    Returns:
        FinancialQuote with calculated monthly installment and totals
    """
    if principal <= 0:
        raise ValueError("Principal must be positive")
    if annual_interest_rate < 0:
        raise ValueError("Interest rate cannot be negative")
    if tenure_months <= 0:
        raise ValueError("Tenure must be positive")

    # Handle zero interest rate (no interest loan)
    if annual_interest_rate == 0:
        monthly_installment = principal / tenure_months
        return FinancialQuote(
            principal=principal,
            interest_rate=annual_interest_rate,
            tenure_months=tenure_months,
            monthly_installment=round(monthly_installment, 2),
            total_interest=0.0,
            total_amount=principal,
        )

    # Monthly interest rate
    r = annual_interest_rate / 12
    n = tenure_months

    # PMT formula: P * [r(1+r)^n] / [(1+r)^n - 1]
    compound = (1 + r) ** n
    monthly_installment = principal * (r * compound) / (compound - 1)

    # Calculate totals
    total_amount = monthly_installment * tenure_months
    total_interest = total_amount - principal

    return FinancialQuote(
        principal=round(principal, 2),
        interest_rate=annual_interest_rate,
        tenure_months=tenure_months,
        monthly_installment=round(monthly_installment, 2),
        total_interest=round(total_interest, 2),
        total_amount=round(total_amount, 2),
    )


def check_affordability(
    monthly_installment: float,
    monthly_income: float,
    max_debt_burden_ratio: float = 0.5,
) -> tuple[bool, float]:
    """
    Check if the monthly installment is affordable based on debt burden ratio.

    Args:
        monthly_installment: Calculated monthly payment
        monthly_income: User's monthly income
        max_debt_burden_ratio: Maximum allowed DBR (default 50%)

    Returns:
        Tuple of (is_affordable, actual_dbr)
    """
    if monthly_income <= 0:
        return False, 1.0

    actual_dbr = monthly_installment / monthly_income
    is_affordable = actual_dbr <= max_debt_burden_ratio

    return is_affordable, round(actual_dbr, 4)
