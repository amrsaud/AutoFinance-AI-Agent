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
Confirmation node for human-in-the-loop search approval.
Uses LLM to classify if user confirmed or cancelled the search.
"""

import logging
from enum import Enum

from langchain_core.messages import AIMessage, SystemMessage
from opentelemetry import trace
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class ConfirmationIntent(str, Enum):
    """User's intent regarding the search confirmation."""

    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    UNCLEAR = "unclear"


class ConfirmationResult(BaseModel):
    """Structured output for confirmation classification."""

    intent: ConfirmationIntent = Field(
        ..., description="The user's intent: confirmed, cancelled, or unclear"
    )
    reasoning: str = Field(
        default="", description="Brief explanation of why this intent was detected"
    )


CONFIRMATION_SYSTEM_PROMPT = """You are classifying whether the user confirmed or cancelled a pending car search.

Context: The assistant just asked the user if they want to proceed with a car search. The user responded.

Classify the user's response into ONE of these intents:
- "confirmed": User wants to proceed (e.g., "yes", "go ahead", "search", "do it", "sure", "ok", "proceed", "start")
- "cancelled": User does NOT want to proceed (e.g., "no", "cancel", "stop", "never mind", "don't", "wait")
- "unclear": User's intent is ambiguous or they're asking something else

Be generous with confirmation - if the user seems to want to proceed, classify as confirmed."""


async def check_confirmation(state: dict, llm) -> dict:
    """Check if user confirmed or cancelled the search using LLM classification.

    Args:
        state: The current agent state with messages.
        llm: The language model to use for classification.

    Returns:
        dict: Updated state with confirmation status.
    """
    with tracer.start_as_current_span("check_confirmation") as span:
        messages = state.get("messages", [])

        if not messages:
            span.set_attribute("node.output.intent", "unclear")
            return {"search_confirmed": False}

        last_message = messages[-1]
        span.set_attribute("node.input", last_message.content[:100])

        # Use LLM with structured output to classify intent
        structured_llm = llm.with_structured_output(ConfirmationResult)

        try:
            result = await structured_llm.ainvoke(
                [SystemMessage(content=CONFIRMATION_SYSTEM_PROMPT), last_message]
            )

            intent = result.intent
            span.set_attribute("node.output.intent", intent.value)
            span.set_attribute("node.output.reasoning", result.reasoning)

            logger.info(
                f"Confirmation classification: {intent.value} - {result.reasoning}"
            )

            if intent == ConfirmationIntent.CONFIRMED:
                return {"search_confirmed": True}
            elif intent == ConfirmationIntent.CANCELLED:
                return {
                    "search_confirmed": False,
                    "search_params": None,  # Clear pending search
                    "messages": [
                        AIMessage(
                            content="No problem! I've cancelled the search. "
                            "Let me know if you'd like to search for a different car or have any questions."
                        )
                    ],
                }
            else:  # UNCLEAR
                return {
                    "search_confirmed": False,
                    "messages": [
                        AIMessage(
                            content="I'm not sure if you want me to proceed with the search. "
                            "Please say 'yes' to search or 'no' to cancel."
                        )
                    ],
                }

        except Exception as e:
            logger.error(f"Error classifying confirmation: {e}")
            span.set_attribute("node.output.error", str(e))
            # Default to unclear on error
            return {
                "search_confirmed": False,
                "messages": [
                    AIMessage(
                        content="I couldn't understand your response. "
                        "Would you like me to proceed with the search? Please say 'yes' or 'no'."
                    )
                ],
            }


def should_execute_search(state: dict) -> str:
    """Determine if search should be executed based on confirmation.

    Args:
        state: The current agent state.

    Returns:
        str: "market_search" if confirmed, "respond" otherwise.
    """
    search_confirmed = state.get("search_confirmed", False)
    search_params = state.get("search_params")

    if search_params and search_confirmed:
        return "market_search"
    else:
        return "respond"
