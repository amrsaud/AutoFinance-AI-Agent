import logging
from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from opentelemetry import trace
from pydantic import BaseModel, Field

from models import AgentState
from tools.supabase_client import submit_application

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


async def submission_node(state: dict, llm) -> dict:
    """Handles loan application submission via LLM intent detection."""
    with tracer.start_as_current_span("submission_node") as span:
        messages = state.get("messages", [])
        last_message = messages[-1].content if messages else ""

        quotes = state.get("generated_quotes") or []
        user_profile = state.get("user_profile")
        vehicle = state.get("selected_vehicle")

        # 1. Use LLM to Detect User Decision
        class UserDecision(BaseModel):
            """User's decision on the financing options presented."""

            decision: Literal["select_option", "start_over", "unknown"] = Field(
                ...,
                description="The user's intent: select a specific option, start over/cancel, or unknown/unclear.",
            )
            option_number: Optional[int] = Field(
                None,
                description="The option number selected (1-based index), if applicable.",
            )

        system_prompt = """You are a helpful assistant validating user decisions on financing options.
        The user has been presented with a list of numbered loan options (e.g., Option 1, Option 2, etc.).
        
        Analyze the user's response:
        - If they select an option (e.g., "Option 1", "I'll take the first one", "Confirm 2"), output 'select_option' and the option_number.
        - If they want to cancel, reset, or look for another car (e.g., "No", "Start over", "Cancel", "Reject"), output 'start_over'.
        - If the response is unrelated or unclear, output 'unknown'.
        """

        try:
            structured_llm = llm.with_structured_output(UserDecision)
            decision_result = await structured_llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=last_message),
                ]
            )
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            # Fallback
            decision_result = UserDecision(decision="unknown")

        # 2. Handle Decisions
        if decision_result.decision == "start_over":
            return {
                "messages": [
                    AIMessage(
                        content="Okay, let's start over. What kind of car are you looking for?"
                    )
                ],
                "search_params": None,
                "search_results": [],
                "search_confirmed": False,
                "selected_vehicle": None,
                "user_profile": None,
                "eligible_policies": None,
                "generated_quotes": None,
                "awaiting_submission": False,
                "_next_node": None,
            }

        elif decision_result.decision == "select_option":
            selection_index = (
                (decision_result.option_number - 1)
                if decision_result.option_number
                else -1
            )

            selected_quote = None
            if 0 <= selection_index < len(quotes):
                selected_quote = quotes[selection_index]
            else:
                return {
                    "messages": [
                        AIMessage(
                            content=f"Invalid option number. Please choose a number between 1 and {len(quotes)}."
                        )
                    ]
                }

            # Submit to Supabase
            session_id = f"session_{user_profile.contact_phone}"
            try:
                profile_dict = user_profile.model_dump()
                vehicle_dict = vehicle.model_dump()
                quote_dict = selected_quote.model_dump()

                app_id = submit_application.invoke(
                    {
                        "session_id": session_id,
                        "user_profile": profile_dict,
                        "vehicle_details": vehicle_dict,
                        "selected_quote": quote_dict,
                    }
                )

                confirmation_msg = (
                    f"🎉 **Application Submitted Successfully!**\n\n"
                    f"Your Request ID for detailed review is: `{app_id}`\n\n"
                    f"We have received your application for the **{vehicle.make} {vehicle.model}** with "
                    f"**{selected_quote.plan_name}** financing.\n"
                    "Our team will contact you shortly to finalize the paperwork.\n\n"
                    "This session will now reset. Feel free to search for another vehicle anytime!"
                )

                # Hard Reset State
                return {
                    "messages": [AIMessage(content=confirmation_msg)],
                    "search_params": None,
                    "search_results": [],
                    "search_confirmed": False,
                    "selected_vehicle": None,
                    "user_profile": None,
                    "eligible_policies": None,
                    "generated_quotes": None,
                    "awaiting_submission": False,
                    "_next_node": None,
                }
            except Exception as submit_err:
                logger.error(f"Submission Error: {submit_err}")
                return {
                    "messages": [
                        AIMessage(
                            content="I encountered an error submitting your application. Please try again."
                        )
                    ]
                }

        else:  # Unknown
            return {
                "messages": [
                    AIMessage(
                        content="Okay, let's start over. What kind of car are you looking for?"
                    )
                ],
                "search_params": None,
                "search_results": [],
                "search_confirmed": False,
                "selected_vehicle": None,
                "user_profile": None,
                "eligible_policies": None,
                "generated_quotes": None,
                "awaiting_submission": False,
                "_next_node": None,
            }
