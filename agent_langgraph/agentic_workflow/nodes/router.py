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
Router Node - Entry point for the AutoFinance workflow.

Analyzes incoming messages and routes to appropriate nodes based on:
- New request: Start vehicle search flow
- Status check: Query existing application
- Continue session: Resume active workflow phase
"""

import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END
from models import AgentState, WorkflowPhase


def _is_status_check(message: str) -> bool:
    """Check if user is asking about application status."""
    status_patterns = [
        r"status",
        r"check.*(?:application|request)",
        r"(?:application|request).*(?:id|number)",
        r"track",
        r"where.*(?:application|request)",
        r"update.*(?:application|request)",
    ]
    message_lower = message.lower()
    return any(re.search(pattern, message_lower) for pattern in status_patterns)


def _extract_request_id(message: str) -> str | None:
    """Extract UUID request ID from message if present."""
    uuid_pattern = (
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    )
    match = re.search(uuid_pattern, message)
    return match.group(0) if match else None


def _is_confirmation(message: str) -> bool:
    """Check if message is a positive confirmation."""
    confirm_patterns = [
        r"^(?:yes|yeah|yep|yup|sure|ok|okay|confirm|proceed|go ahead)\b",
        r"^(?:نعم|أيوه|تمام|موافق)\b",  # Arabic confirmations
    ]
    message_lower = message.lower().strip()
    return any(re.match(pattern, message_lower) for pattern in confirm_patterns)


def _is_rejection(message: str) -> bool:
    """Check if message is a negative response."""
    reject_patterns = [
        r"^(?:no|nope|cancel|stop|never mind|nevermind)\b",
        r"^(?:لا|الغي|توقف)\b",  # Arabic rejections
    ]
    message_lower = message.lower().strip()
    return any(re.match(pattern, message_lower) for pattern in reject_patterns)


def _is_closing_intent(message: str) -> bool:
    """Check if user is trying to end the conversation."""
    closing_patterns = [
        r"^(?:thanks|thank you|bye|goodbye|done|finished)\b",
        r"^(?:شكرا|مع السلامة)\b",
    ]
    message_lower = message.lower().strip()
    return any(re.match(pattern, message_lower) for pattern in closing_patterns)


def router_node(state: AgentState) -> dict:
    """
    Entry point that analyzes the incoming message and routes accordingly.

    This node does not modify state significantly - it just determines
    routing via the conditional edge function.
    """
    # Check for zombie prevention: if request_id exists, workflow is complete
    if state.request_id:
        last_message = state.messages[-1] if state.messages else None
        if isinstance(last_message, HumanMessage):
            user_text = last_message.content

            # Handle closing intent after submission
            if _is_closing_intent(str(user_text)):
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "Thank you for using AutoFinance! Your application "
                                f"(Request ID: {state.request_id}) has been submitted. "
                                "Our team will review it and contact you soon. Have a great day! 🚗"
                            )
                        )
                    ]
                }

            # Handle status check after submission
            if _is_status_check(str(user_text)):
                return {}  # Route to status check

    # For new conversations, show welcome message
    if state.current_phase == WorkflowPhase.ONBOARDING and len(state.messages) <= 1:
        welcome_message = """
# Welcome to AutoFinance AI! 🚗💰

I'm your Financial Co-Pilot, here to help you find your dream car and calculate 
an affordable loan in Egypt.

**What I can do:**
- 🔍 Search car listings from Hatla2ee & Dubizzle
- 📊 Calculate monthly installments based on your income
- 📝 Submit pre-approval requests for back-office review

**How can I help you today?**

1. **Start New Request** - Find a car and calculate financing
2. **Check Status** - Look up an existing application using your Request ID

Just tell me what car you're looking for (e.g., "I want a 2022 Hyundai Tucson") 
or provide your Request ID to check status.
"""
        return {
            "messages": [AIMessage(content=welcome_message)],
            "current_phase": WorkflowPhase.DISCOVERY,
        }

    return {}


def route_initial(
    state: AgentState,
) -> Literal["status_check", "search_param", "router", "continue_flow", "__end__"]:
    """
    Conditional routing function for the entry point.

    Routes to:
    - status_check: User asking about application status
    - search_param: New search request
    - continue_flow: Resume active workflow
    - __end__: Workflow complete or closing intent
    """
    if not state.messages:
        return "router"

    last_message = state.messages[-1]
    if not isinstance(last_message, HumanMessage):
        return "continue_flow"

    user_text = str(last_message.content)

    # Zombie prevention: workflow complete
    if state.request_id and _is_closing_intent(user_text):
        return END

    # Status check request
    if _is_status_check(user_text):
        return "status_check"

    # Confirmation handling during awaiting states
    if state.awaiting_confirmation:
        if _is_confirmation(user_text):
            if state.awaiting_confirmation == "search":
                return "market_search"
            elif state.awaiting_confirmation == "quote":
                return "submission"
        elif _is_rejection(user_text):
            return "search_param"  # Restart search

    # Continue based on current phase
    phase_routing = {
        WorkflowPhase.ONBOARDING: "router",
        WorkflowPhase.DISCOVERY: "search_param",
        WorkflowPhase.PROFILING: "profiling_logic",
        WorkflowPhase.QUOTATION: "quotation",
        WorkflowPhase.SUBMISSION: "submission",
        WorkflowPhase.COMPLETED: END,
    }

    return phase_routing.get(state.current_phase, "router")
