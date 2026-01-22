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
Credit Policy RAG tool using DataRobot VectorDB.

Retrieves applicable credit policies based on user profile and vehicle details.
For MVP, includes fallback policies when VectorDB is not configured.
"""

from datetime import datetime
from typing import Optional

from models import CreditPolicy, EmploymentType, Vehicle

# Fallback policy grid for MVP when VectorDB is not configured
# These represent sample internal credit policies
FALLBACK_POLICIES = {
    EmploymentType.CORPORATE: {
        "interest_rate": 0.16,  # 16% annual
        "max_tenure_months": 84,
        "max_debt_burden_ratio": 0.55,
        "min_income": 8000.0,
        "max_vehicle_age": 10,
    },
    EmploymentType.SALARIED: {
        "interest_rate": 0.18,  # 18% annual
        "max_tenure_months": 72,
        "max_debt_burden_ratio": 0.50,
        "min_income": 6000.0,
        "max_vehicle_age": 10,
    },
    EmploymentType.SELF_EMPLOYED: {
        "interest_rate": 0.20,  # 20% annual
        "max_tenure_months": 60,
        "max_debt_burden_ratio": 0.45,
        "min_income": 10000.0,
        "max_vehicle_age": 8,
    },
    EmploymentType.OTHER: {
        "interest_rate": 0.22,  # 22% annual
        "max_tenure_months": 48,
        "max_debt_burden_ratio": 0.40,
        "min_income": 15000.0,
        "max_vehicle_age": 7,
    },
}


def get_vehicle_age(vehicle: Vehicle) -> int:
    """Calculate the age of a vehicle in years."""
    current_year = datetime.now().year
    return current_year - vehicle.year


def retrieve_credit_policy(
    employment_type: EmploymentType,
    monthly_income: float,
    vehicle: Vehicle,
    vectordb_id: Optional[str] = None,
) -> CreditPolicy:
    """
    Retrieve the applicable credit policy for a user.

    In a full implementation, this would query the DataRobot VectorDB
    with the user profile and vehicle details. For MVP, it uses
    fallback policies based on employment type.

    Args:
        employment_type: User's employment category
        monthly_income: User's monthly income in EGP
        vehicle: Selected vehicle
        vectordb_id: Optional DataRobot VectorDB ID for RAG

    Returns:
        CreditPolicy with eligibility status
    """
    # TODO: Implement DataRobot VectorDB RAG when vectordb_id is provided
    # For now, use fallback policies

    policy_params = FALLBACK_POLICIES.get(
        employment_type, FALLBACK_POLICIES[EmploymentType.OTHER]
    )

    # Check eligibility criteria
    vehicle_age = get_vehicle_age(vehicle)
    is_eligible = True
    rejection_reason = None

    # Check income minimum
    if monthly_income < policy_params["min_income"]:
        is_eligible = False
        rejection_reason = (
            f"Minimum income requirement is {policy_params['min_income']:,.0f} EGP. "
            f"Your income of {monthly_income:,.0f} EGP does not meet this threshold."
        )

    # Check vehicle age
    elif vehicle_age > policy_params["max_vehicle_age"]:
        is_eligible = False
        rejection_reason = (
            f"Maximum vehicle age allowed is {policy_params['max_vehicle_age']} years. "
            f"The selected vehicle ({vehicle.year}) is {vehicle_age} years old."
        )

    return CreditPolicy(
        interest_rate=policy_params["interest_rate"],
        max_tenure_months=policy_params["max_tenure_months"],
        max_debt_burden_ratio=policy_params["max_debt_burden_ratio"],
        min_income=policy_params["min_income"],
        max_vehicle_age=policy_params["max_vehicle_age"],
        is_eligible=is_eligible,
        rejection_reason=rejection_reason,
    )


def format_policy_summary(policy: CreditPolicy) -> str:
    """Format credit policy for display to user."""
    if not policy.is_eligible:
        return f"""
## ❌ Eligibility Check Failed

{policy.rejection_reason}

Unfortunately, we cannot proceed with this loan application. 
You may:
- Try a different vehicle that meets our criteria
- Contact our support for alternative financing options
"""

    return f"""
## ✅ You Are Eligible!

Based on your profile, here are the applicable terms:

- **Interest Rate:** {policy.interest_rate * 100:.1f}% per annum
- **Maximum Tenure:** {policy.max_tenure_months} months
- **Maximum Debt Burden:** {policy.max_debt_burden_ratio * 100:.0f}% of income
"""
