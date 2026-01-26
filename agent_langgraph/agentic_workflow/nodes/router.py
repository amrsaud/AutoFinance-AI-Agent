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
Router node for intent classification.
Classifies user intent as 'search', 'reset', or 'chat'.
"""

import logging

from langchain_core.messages import SystemMessage
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

ROUTER_SYSTEM_PROMPT = """You are a strict intent classifier. Reply with ONLY one word: 'search', 'reset', 'select', or 'chat'.

IMPORTANT Rules:
1. 'search': User wants to find/buy cars. Mentions BRAND (Toyota), MODEL (Corolla), PRICE, or "Looking for...".
2. 'select': User wants to choose a car from a list. e.g., "Option 1", "The first one", "I'll take the BMW", "Select car 2".
3. 'reset': "Start over", "Clear", "New search".
4. 'chat': Greetings, "Thanks", "Hello". Or providing information like "My salary is 5000", "I am a freelancer".

Examples:
- "I want a Hyundai Tucson" -> search
- "Option 2" -> select
- "I choose the first one" -> select
- "I like the red one" -> select
- "My income is 10000" -> chat (This is info provided during profiling)
- "Freelancer" -> chat
- "Start over" -> reset
- "Hello" -> chat

Reply with exactly one word."""


async def route_intent(state: dict, llm) -> str:
    """Classify user intent and route based on state.

    Args:
        state: The current agent state.
        llm: The language model.

    Returns:
        str: Next node name.
    """
    with tracer.start_as_current_span("route_intent") as span:
        messages = state.get("messages", [])
        if not messages:
            return "respond"

        # Check for state-based overrides (Financing Flow)
        selected_vehicle = state.get("selected_vehicle")
        user_profile = state.get("user_profile")

        # Determine intent of NEW message
        recent_messages = messages[-5:]
        response = await llm.ainvoke(
            [SystemMessage(content=ROUTER_SYSTEM_PROMPT)] + recent_messages
        )
        intent = response.content.strip().lower()
        span.set_attribute("node.output.intent", intent)
        logger.info(
            f"Router Intent: {intent} | Vehicle Set: {bool(selected_vehicle)} | Profile Set: {bool(user_profile)}"
        )

        # 1. Global Exits
        if "reset" in intent:
            return "reset"

        # 0. Check for Submission Intent (High Priority State)
        if state.get("awaiting_submission"):
            return "submission"

        if "search" in intent:
            # If user wants to search again, allow it (clears selection in search_param/market_search implicit?)
            # Ideally search_param should clear old selection.
            # We will let 'search_param' handle new query.
            return "search_param"

        # 2. Financing / Profiling Flow

        # If user explicitly selects a vehicle, go to profiling (which handles selection)
        if "select" in intent:
            return "profiling"

        # If we are already in the financing loop (Vehicle Selected)
        if selected_vehicle:
            # If profile is fully complete (object exists and fields valid? - Assume object existence = complete for routing,
            # but profiling node ensures completeness before creating object.
            # Wait, our `profiling` node only creates `UserProfile` object when ALL fields present.
            # So if `user_profile` is NOT None, it is Complete.
            # Check if profile is TRULY complete (all required fields present)
            # user_profile might be a partial object from profiling node
            is_complete = (
                user_profile
                and user_profile.monthly_income is not None
                and user_profile.employment_type is not None
                and user_profile.existing_debt_obligations is not None
                and user_profile.contact_name
                and user_profile.contact_phone
            )

            if is_complete:
                return "financing"
            else:
                # Profile incomplete, any "chat" (providing info) goes to profiling
                return "profiling"

        # 3. Default Fallback
        return "respond"
