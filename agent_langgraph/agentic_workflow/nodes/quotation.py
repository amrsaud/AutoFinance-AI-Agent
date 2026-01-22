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
Quotation Node - Calculate and present loan quote.

Uses the PMT formula to calculate monthly installment and
presents a summary card for user confirmation.
"""

from typing import Any

from config import Config
from langchain_core.messages import AIMessage
from models import AgentState, WorkflowPhase
from tools.installment_calculator import (
    calculate_monthly_installment,
    check_affordability,
)

config = Config()


def quotation_node(state: AgentState) -> dict[str, Any]:
    """
    Calculate loan quote and present to user for confirmation.

    This is a human-in-the-loop checkpoint - the user must confirm
    before proceeding to submission.
    """
    # Validate required data
    if not state.selected_vehicle:
        return {
            "messages": [AIMessage(content="⚠️ No vehicle selected.")],
            "current_phase": WorkflowPhase.DISCOVERY,
        }

    if not state.applicable_policy:
        return {
            "messages": [AIMessage(content="⚠️ No credit policy retrieved.")],
            "current_phase": WorkflowPhase.PROFILING,
        }

    vehicle = state.selected_vehicle
    policy = state.applicable_policy
    monthly_income = state.monthly_income or 0

    # Calculate loan details using PMT formula
    tenure_months = min(
        policy.max_tenure_months,
        config.default_loan_tenure_months,
    )

    quote = calculate_monthly_installment(
        principal=vehicle.price,
        annual_interest_rate=policy.interest_rate,
        tenure_months=tenure_months,
    )

    # Check affordability
    is_affordable, dbr = check_affordability(
        monthly_installment=quote.monthly_installment,
        monthly_income=monthly_income,
        max_debt_burden_ratio=policy.max_debt_burden_ratio,
    )

    if not is_affordable:
        # Suggest longer tenure or different vehicle
        message = f"""
## ⚠️ Affordability Concern

Based on your income of **{monthly_income:,.0f} EGP**, the calculated monthly installment 
of **{quote.monthly_installment:,.0f} EGP** represents **{dbr * 100:.1f}%** of your income.

Our policy allows a maximum debt burden of **{policy.max_debt_burden_ratio * 100:.0f}%**.

**Suggestions:**
- Consider a longer loan tenure to reduce monthly payments
- Look for a less expensive vehicle
- Provide additional proof of income if you have other sources

Would you like to:
1. **Continue anyway** (with a note that approval may require review)
2. **Search for a different vehicle**
"""
        return {
            "messages": [AIMessage(content=message)],
            "financial_quote": quote,
            "current_phase": WorkflowPhase.QUOTATION,
        }

    # Format quote summary card
    summary_message = _format_quote_summary(
        vehicle=vehicle,
        quote=quote,
        monthly_income=monthly_income,
        employment_type=state.employment_type,
        dbr=dbr,
    )

    return {
        "messages": [AIMessage(content=summary_message)],
        "financial_quote": quote,
        "awaiting_confirmation": "quote",
        "current_phase": WorkflowPhase.QUOTATION,
    }


def _format_quote_summary(vehicle, quote, monthly_income, employment_type, dbr) -> str:
    """Format the complete quote summary for user review."""

    return f"""
## 📋 Pre-Approval Estimate

### Vehicle Details
| Item | Details |
|------|---------|
| **Vehicle** | {vehicle.name} |
| **Year** | {vehicle.year} |
| **Price** | {vehicle.price:,.0f} EGP |
| **Source** | [{vehicle.source_site}]({vehicle.source_url}) |

---

### Loan Details
| Item | Details |
|------|---------|
| **Principal** | {quote.principal:,.0f} EGP |
| **Interest Rate** | {quote.interest_rate * 100:.1f}% p.a. |
| **Tenure** | {quote.tenure_months} months |
| **Monthly Installment** | **{quote.monthly_installment:,.0f} EGP** |
| **Total Interest** | {quote.total_interest:,.0f} EGP |
| **Total Amount Payable** | {quote.total_amount:,.0f} EGP |

---

### Your Profile
| Item | Details |
|------|---------|
| **Monthly Income** | {monthly_income:,.0f} EGP |
| **Employment Type** | {employment_type.value.replace("_", " ").title() if employment_type else "N/A"} |
| **Debt Burden Ratio** | {dbr * 100:.1f}% ✅ |

---

## ✅ You're Ready to Apply!

This is a **pre-approval estimate**. Final approval is subject to document verification 
by our back-office team.

**Would you like to proceed with this request?**

Reply **"Yes"** to continue to the application form, or **"No"** to make changes.
"""
