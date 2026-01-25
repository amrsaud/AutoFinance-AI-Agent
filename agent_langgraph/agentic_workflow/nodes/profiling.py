import logging

# from typing import List, Optional # Removed unused import
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from opentelemetry import trace
from pydantic import BaseModel, Field

from models import EmploymentType, UserProfile

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class UserProfileExtraction(BaseModel):
    """Extraction model for user profile fields (all optional)."""

    monthly_income: Optional[float] = Field(None, description="Monthly income in EGP")
    employment_type: Optional[EmploymentType] = Field(
        None, description="Employment category matching credit policies"
    )
    existing_debt_obligations: Optional[float] = Field(
        None, description="Total existing monthly debt payments in EGP"
    )
    contact_name: Optional[str] = Field(None, description="Full name")
    contact_phone: Optional[str] = Field(None, description="Phone number")
    contact_email: Optional[str] = Field(None, description="Email address")


PROFILING_SYSTEM_PROMPT = """You are a helpful financial assistant collecting information for a vehicle loan application.
Your goal is to extract user information to complete their profile.
If the user provides information, extract it.
Do not make up information.
"""


async def profiling_node(state: dict, llm) -> dict:
    """Collects and merges user profile information.

    Args:
        state: Current graph state.
        llm: Language model.

    Returns:
        dict: Update to state (messages or user_profile).
    """
    with tracer.start_as_current_span("profiling_node") as span:
        # 1. Get current profile data (if any)
        current_profile_obj = state.get("user_profile")
        # Initialize dictionary from existing object or empty
        if current_profile_obj:
            # If it's a Pydantic object, dict() it. If dict, use as is.
            # State management might convert pydantic to dict automatically in some LangGraph setups?
            # Safer to check.
            if isinstance(current_profile_obj, BaseModel):
                profile_data = current_profile_obj.model_dump()
            else:
                profile_data = current_profile_obj.copy()
        else:
            profile_data = {}

        # 2. Extract new info from the last user message
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], HumanMessage):
            last_message = messages[-1].content

            try:
                structured_llm = llm.with_structured_output(UserProfileExtraction)
                extraction = await structured_llm.ainvoke(
                    [
                        SystemMessage(content=PROFILING_SYSTEM_PROMPT),
                        HumanMessage(content=last_message),
                    ]
                )

                # Merge extracted fields (only if not None)
                extraction_dict = extraction.model_dump(
                    exclude_unset=True, exclude_none=True
                )
                if extraction_dict:
                    logger.info(f"Extracted profile data: {extraction_dict.keys()}")
                    profile_data.update(extraction_dict)

            except Exception as e:
                logger.error(f"Profiling extraction failed: {e}")

        # 3. Check for missing required fields
        required_fields = [
            "monthly_income",
            "employment_type",
            "existing_debt_obligations",
            "contact_name",
            "contact_phone",
            "contact_email",
        ]

        missing = [
            f
            for f in required_fields
            if f not in profile_data or profile_data[f] is None
        ]

        # 4. If missing fields, ask user
        # 4. If missing fields, ask user
        if missing:
            missing_clean = [f.replace("_", " ") for f in missing]

            msg_content = f"Could you please provide your **{missing_clean[0]}**?"
            if len(missing) > 1:
                msg_content = f"To proceed, I need: {', '.join(missing_clean)}."

            # Save partial state
            # models.py now allows optional fields, so this is safe
            partial_profile = UserProfile(**profile_data)

            return {
                "messages": [AIMessage(content=msg_content)],
                "user_profile": partial_profile,
            }

        # 5. If complete
        complete_profile = UserProfile(**profile_data)
        return {
            "user_profile": complete_profile
            # No message, proceed directly to financing
        }
