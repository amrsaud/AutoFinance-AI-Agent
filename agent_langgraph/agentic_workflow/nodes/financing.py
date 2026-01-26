import logging
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from opentelemetry import trace
from pydantic import BaseModel, Field

from models import CreditPolicy, LoanQuote
from tools.policy_retriever import retrieve_eligible_policies
from tools.calculator import calculate_loan_options

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


async def financing_node(state: dict, llm) -> dict:
    """Retrieves policies and calculates quotes for the user.

    Args:
        state: Current graph state (must have full user_profile and selected_vehicle).
        llm: Language model (used for formatting if needed, though structure handled here).

    Returns:
        dict: Update to state with policies, quotes, and response message.
    """
    with tracer.start_as_current_span("financing_node") as span:
        profile = state.get("user_profile")
        vehicle = state.get("selected_vehicle")

        # Safety check (Router should prevent this, but good practice)
        if not profile or not vehicle:
            return {
                "messages": [AIMessage(content="Error: Profile or Vehicle missing.")]
            }

        # 1. Retrieve Eligible Policies
        # Tool expects primitives
        # Handle potential string vs Enum type
        emp_type = profile.employment_type
        emp_value = emp_type.value if hasattr(emp_type, "value") else str(emp_type)

        # Safety check for income
        income = profile.monthly_income if profile.monthly_income is not None else 0.0

        # 1. Retrieve Eligible Policies (now returns list of strings)
        raw_policies_data = retrieve_eligible_policies.invoke(
            {
                "min_income": income,
                "employment_category": emp_value,
            }
        )

        if not raw_policies_data or (
            isinstance(raw_policies_data, list) and not raw_policies_data
        ):
            logger.warning(f"No policies found: {raw_policies_data}")
            return {
                "messages": [
                    AIMessage(
                        content="I couldn't find any credit policies matching your profile exactly. You might need a guarantor."
                    )
                ]
            }

        # 2. Use LLM to Structure and Filter Policies
        class PolicyExtraction(BaseModel):
            """Extracted credit policies matching the user profile."""

            valid_policies: list[CreditPolicy] = Field(
                ...,
                description="List of policies that match the user's employment and income.",
            )

        extraction_system_prompt = """You are a Credit Policy Analyst.
        Your task is to analyze the provided raw policy documents and extract structured CreditPolicy objects.
        
        CRITICAL: Only extract policies that strictly match the User's attributes:
        1. Employment Category: Must match user's job type.
        2. Income: User's income must be >= policy minimum income.
        
        User Profile:
        - Income: {income} EGP
        - Employment: {emp_value}

        Return a list of valid CreditPolicy objects.
        If a field (like Max DBR) is missing, use reasonable defaults (DBR=0.50, Rate=15.0).
        """

        try:
            structured_retriever = llm.with_structured_output(PolicyExtraction)
            extraction_result = await structured_retriever.ainvoke(
                [
                    SystemMessage(
                        content=extraction_system_prompt.format(
                            income=income, emp_value=emp_value
                        )
                    ),
                    HumanMessage(
                        content=f"Raw Policy Documents:\n{str(raw_policies_data)}"
                    ),
                ]
            )

            policy_objects = extraction_result.valid_policies
        except Exception as e:
            logger.error(f"Policy extraction failed: {e}")
            policy_objects = []

        if not policy_objects:
            return {
                "messages": [
                    AIMessage(
                        content="I found some policies but none seem to perfectly match your profile criteria."
                    )
                ]
            }

        # Deduplicate policies based on description + rate + tenure
        unique_policies = {}
        for p in policy_objects:
            # Create a unique key. Note: description itself might be minimal, so include rates.
            key = (p.description, p.interest_rate, p.max_tenure_months)
            if key not in unique_policies:
                unique_policies[key] = p

        policy_objects = list(unique_policies.values())

        # 3. Calculate Quotes via Tool
        # Convert objects back to dicts for Calculator Tool
        # Calculator expects dicts with keys: 'interest_rate', 'max_dbr', 'max_tenure_months'
        policies_dicts = [p.model_dump() for p in policy_objects]

        # Vehicle price logic (use vehicle.price or fallback)
        price = vehicle.price if vehicle.price else 1000000.0

        quotes = calculate_loan_options.invoke(
            {
                "vehicle_price": float(price),
                "user_income": income,
                "existing_debt": profile.existing_debt_obligations,
                "policies": policies_dicts,
            }
        )

        # 4. Format Response (Static Text)
        affordable_quotes = [q for q in quotes if q.get("is_affordable", False)]

        response_text = ""
        if not affordable_quotes:
            response_text = (
                f"Based on your income (`{income} EGP`) and existing debts, "
                "the monthly installments for this vehicle exceed the allowed **Debt Burden Ratio (DBR)** limits.\n\n"
                "**Suggestions:**\n"
                "- Try a longer tenure if available.\n"
                "- Choose a lower-priced vehicle.\n"
                "- Increase your downpayment.\n\n"
                'Please reply with **"Start Over"** to search for other cars.'
            )
        else:
            response_text = (
                f"I found **{len(affordable_quotes)}** financing options for the **{vehicle.make} {vehicle.model}** "
                f"({vehicle.price:,.0f} EGP):\n\n"
            )

            for i, q in enumerate(affordable_quotes, 1):
                plan_name = q.get("plan_name", f"Option {i}")
                inst = q.get("monthly_installment", 0)
                dp = q.get("downpayment", 0)
                rate = q.get("interest_rate", 0)
                # Calculator tool returns 'tenure_months'
                tenure = q.get("tenure_months", q.get("tenure", 0))

                response_text += (
                    f"**{i}. {plan_name}**\n"
                    f"- Monthly Installment: **{inst:,.0f} EGP**\n"
                    f"- Downpayment (20%): {dp:,.0f} EGP\n"
                    f"- Interest Rate: {rate}%\n"
                    f"- Tenure: {tenure} months\n\n"
                )

            response_text += (
                'To proceed, please reply with the **Option Name** (e.g., *"Option 1"*) '
                'or say **"Start Over"** to look for another car.'
            )

        response_msg = AIMessage(content=response_text)

        # Convert dictionaries to LoanQuote objects for state persistence
        # Provide defaults for missing fields if any
        quote_objects = []
        for q in quotes:
            if "plan_name" in q:
                # Ensure tenure_months is set for model validation
                if "tenure" in q and "tenure_months" not in q:
                    q["tenure_months"] = q["tenure"]
                quote_objects.append(LoanQuote(**q))

        return {
            "eligible_policies": policy_objects,
            "generated_quotes": quote_objects,
            "awaiting_submission": True if policy_objects and quote_objects else False,
            "messages": [response_msg],
        }
