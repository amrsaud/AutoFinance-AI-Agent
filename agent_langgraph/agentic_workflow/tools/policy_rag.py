# Copyright 2025 DataRobot, Inc.
# Licensed under the Apache License, Version 2.0
"""Credit policy RAG tool."""

from datetime import datetime

from langchain_core.tools import tool

try:
    from config import Config
    from state import CreditPolicy, Vehicle
except ImportError:
    from ..config import Config
    from ..state import CreditPolicy, Vehicle


@tool
def get_credit_policy(
    vehicle: Vehicle,
    monthly_income: float,
    employment_type: str,
) -> CreditPolicy:
    """Retrieve applicable credit policy for vehicle and customer profile.

    Args:
        vehicle: Selected vehicle
        monthly_income: Customer's monthly income in EGP
        employment_type: 'salaried', 'self_employed', or 'corporate'
    """
    config = Config()
    current_year = datetime.now().year
    vehicle_age = current_year - vehicle.year

    # Check vehicle age
    if vehicle_age > config.max_vehicle_age:
        return CreditPolicy(
            interest_rate=config.default_interest_rate,
            max_dbr=config.max_dbr,
            min_income=config.min_monthly_income,
            max_tenure_months=config.default_tenure_months,
            max_vehicle_age=config.max_vehicle_age,
            eligible=False,
            rejection_reason=f"Vehicle is {vehicle_age} years old, max allowed is {config.max_vehicle_age}",
        )

    # Check income
    if monthly_income < config.min_monthly_income:
        return CreditPolicy(
            interest_rate=config.default_interest_rate,
            max_dbr=config.max_dbr,
            min_income=config.min_monthly_income,
            max_tenure_months=config.default_tenure_months,
            max_vehicle_age=config.max_vehicle_age,
            eligible=False,
            rejection_reason=f"Income {monthly_income:,.0f} EGP below minimum {config.min_monthly_income:,.0f}",
        )

    # Adjust rate by employment
    interest_rate = config.default_interest_rate
    if employment_type.lower() == "corporate":
        interest_rate -= 2.0
    elif employment_type.lower() == "self_employed":
        interest_rate += 2.0

    # Adjust tenure by vehicle age
    max_tenure = config.default_tenure_months
    if vehicle_age > 5:
        max_tenure = min(48, max_tenure)
    if vehicle_age > 7:
        max_tenure = min(36, max_tenure)

    return CreditPolicy(
        interest_rate=interest_rate,
        max_dbr=config.max_dbr,
        min_income=config.min_monthly_income,
        max_tenure_months=max_tenure,
        max_vehicle_age=config.max_vehicle_age,
        eligible=True,
        rejection_reason=None,
    )
