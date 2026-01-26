import logging
import os
from supabase import create_client, Client
from config import Config

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Initialize client globally if possible, or per request
# Using Env vars directly or Config? Config class logic handles env vars.
# We'll use os.getenv as fallback or load Config.
# Since this is a module, let's lazy load.

_supabase_client: Client | None = None


def get_supabase_client() -> Client:
    global _supabase_client
    if _supabase_client:
        return _supabase_client

    config = Config()
    url = config.supabase_url
    key = config.supabase_key

    if not url or not key:
        raise ValueError("Supabase URL and Key must be set in environment variables.")

    _supabase_client = create_client(url, key)
    return _supabase_client


@tool
def submit_application(
    session_id: str, user_profile: dict, vehicle_details: dict, selected_quote: dict
) -> str:
    """Submits a loan application to Supabase.

    Args:
        session_id: Unique session identifier.
        user_profile: User profile dictionary.
        vehicle_details: Vehicle dictionary.
        selected_quote: Quote dictionary.

    Returns:
        str: ID of the created application.
    """
    try:
        client = get_supabase_client()

        # Flatten data for insertion into 'applications' table
        payload = {
            "session_id": session_id,
            # User
            "user_name": user_profile.get("contact_name"),
            "contact_phone": user_profile.get("contact_phone"),
            "contact_email": user_profile.get("contact_email"),
            "monthly_income": user_profile.get("monthly_income"),
            "employment_type": user_profile.get("employment_type")
            if isinstance(user_profile.get("employment_type"), str)
            else str(user_profile.get("employment_type", "")),
            # Vehicle
            "vehicle_make": vehicle_details.get("make"),
            "vehicle_model": vehicle_details.get("model"),
            "vehicle_year": vehicle_details.get("year"),
            "vehicle_price": vehicle_details.get("price"),
            # Quote
            "quote_plan_name": selected_quote.get("plan_name"),
            "quote_monthly_installment": selected_quote.get("monthly_installment"),
            "quote_downpayment": selected_quote.get("downpayment"),
            "quote_tenure": selected_quote.get(
                "tenure_months", selected_quote.get("tenure")
            ),
            "quote_interest_rate": selected_quote.get("interest_rate"),
            "status": "pending_review",
        }

        response = client.table("applications").insert(payload).execute()

        if response.data and len(response.data) > 0:
            return response.data[0]["id"]
        else:
            raise Exception("No data returned from Supabase insert.")

    except Exception as e:
        logger.error(f"Failed to submit application: {e}")
        raise e
