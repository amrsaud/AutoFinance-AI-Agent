from langchain_core.tools import tool
import numpy_financial as npf


@tool
def calculate_loan_options(
    vehicle_price: float, user_income: float, existing_debt: float, policies: list[dict]
) -> list[dict]:
    """
    Calculates loan installments and checks affordability for a list of policies.

    Args:
        vehicle_price: Price of the car in EGP.
        user_income: Monthly income.
        existing_debt: Total existing monthly debt payments.
        policies: List of policy dictionaries (must contain 'interest_rate', 'max_tenure_months', 'max_dbr').

    Returns:
        List of quote dictionaries with installment details and affordability status.
    """
    quotes = []

    # Standard Downpayment assumption (e.g. 20%) -> Principal is 80%
    # If not specified, we can assume 20% downpayment as market standard or calculates on full price.
    # Let's use 20% downpayment as a reasonable default for auto loans in Egypt.
    downpayment_ratio = 0.20
    principal = vehicle_price * (1.0 - downpayment_ratio)

    for policy in policies:
        rate_annual = policy.get("interest_rate", 15.0)
        tenure = policy.get("max_tenure_months", 60)
        max_dbr = policy.get("max_dbr", 0.50)
        policy_id = policy.get("policy_id", "UNKNOWN")

        # Calculate PMT
        # numpy_financial.pmt(rate, nper, pv)
        # rate is per period
        rate_monthly = (rate_annual / 100) / 12

        installment = -1 * npf.pmt(rate_monthly, tenure, principal)

        # Check DBR
        total_obligation = installment + existing_debt
        current_dbr = total_obligation / user_income if user_income > 0 else 1.0

        is_affordable = current_dbr <= max_dbr

        quotes.append(
            {
                "policy_id": policy_id,
                "plan_name": f"{tenure} Months @ {rate_annual}%",
                "monthly_installment": round(installment, 2),
                "tenure_months": tenure,
                "interest_rate": rate_annual,
                "is_affordable": is_affordable,
                "dbr_percentage": round(current_dbr * 100, 2),
                "max_dbr_allowed": round(max_dbr * 100, 2),
                "principal": principal,
                "downpayment": vehicle_price * downpayment_ratio,
            }
        )

    return quotes
