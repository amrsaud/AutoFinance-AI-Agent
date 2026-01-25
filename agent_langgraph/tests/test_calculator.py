import pytest
from agentic_workflow.tools.calculator import calculate_loan_options


def test_calculator_basic():
    """Test basic PMT calculation."""
    # 500,000 car, 20% down -> 400,000 Principal
    # 15% rate, 60 months
    # PMT approx: 400k * (0.0125 / (1 - (1.0125)^-60))
    # 15%/12 = 1.25% = 0.0125
    # roughly 9500 EGP

    policies = [
        {
            "policy_id": "P1",
            "interest_rate": 15.0,
            "max_tenure_months": 60,
            "max_dbr": 0.50,
        }
    ]

    quotes = calculate_loan_options.invoke(
        {
            "vehicle_price": 500000.0,
            "user_income": 30000.0,
            "existing_debt": 0.0,
            "policies": policies,
        }
    )

    assert len(quotes) == 1
    q = quotes[0]
    assert q["principal"] == 400000.0
    # Expected PMT ~ 9515.98
    assert 9500 < q["monthly_installment"] < 9600
    assert q["is_affordable"]
    assert q["dbr_percentage"] < 35.0


def test_calculator_dbr_rejection():
    """Test DBR rejection logic."""
    vehicle_price = 1000000.0  # 800k principal
    # install ~ 19k
    # Income 30k -> 19/30 = 63% -> Reject (Max 50%)

    policies = [
        {
            "policy_id": "P2",
            "interest_rate": 15.0,
            "max_tenure_months": 60,
            "max_dbr": 0.50,
        }
    ]

    quotes = calculate_loan_options.invoke(
        {
            "vehicle_price": vehicle_price,
            "user_income": 30000.0,
            "existing_debt": 0.0,
            "policies": policies,
        }
    )

    # Assert affordable is False (Note: Tool returns bool, so simple assertion works)
    assert not quotes[0]["is_affordable"]
    assert quotes[0]["dbr_percentage"] > 50.0


def test_calculator_existing_debt():
    """Test existing debt subtraction from DBR capacity."""
    # Installment ~ 9500
    # Existing Debt 6000
    # Total 15500
    # Income 30000 -> 15.5/30 = 51.6% -> Reject (Max 50%)

    policies = [
        {
            "policy_id": "P3",
            "interest_rate": 15.0,
            "max_tenure_months": 60,
            "max_dbr": 0.50,
        }
    ]

    quotes = calculate_loan_options.invoke(
        {
            "vehicle_price": 500000.0,
            "user_income": 30000.0,
            "existing_debt": 6000.0,
            "policies": policies,
        }
    )

    assert not quotes[0]["is_affordable"]
    assert quotes[0]["dbr_percentage"] > 50.0
