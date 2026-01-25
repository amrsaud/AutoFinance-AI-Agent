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

ROUTER_SYSTEM_PROMPT = """You are a strict intent classifier. Reply with ONLY one word: 'search', 'reset', or 'chat'.

IMPORTANT: If the user mentions ANY car make, model, year, price, or wants to find/look for/search for a vehicle, ALWAYS return 'search'.

Classifications:
- 'search': User wants to find, search, look for, or get a car. ANY mention of car brands (Toyota, Hyundai, BMW, etc.) or car models (Tucson, Camry, X5) = search
- 'reset': User explicitly wants to start over, reset, clear, or begin fresh
- 'chat': General greetings, capability questions, or non-car related conversation

Examples:
- "I want a Hyundai Tucson" -> search
- "Find me a car under 500k" -> search
- "Looking for a 2024 Toyota" -> search
- "I need a BMW" -> search
- "Show me cars" -> search
- "What Toyota Corolla options are there?" -> search
- "Start over" -> reset
- "Clear everything" -> reset
- "Let's begin again" -> reset
- "What can you do?" -> chat
- "Hello" -> chat
- "Thanks" -> chat

Reply with exactly one word: search, reset, or chat"""


async def route_intent(state: dict, llm) -> str:
    """Classify user intent: 'search', 'reset', or 'chat'.

    Args:
        state: The current agent state containing messages.
        llm: The language model to use for classification.

    Returns:
        str: The next node to route to ('search_param', 'reset', or 'respond').
    """
    with tracer.start_as_current_span("route_intent") as span:
        messages = state.get("messages", [])
        if not messages:
            span.set_attribute("node.output.intent", "chat")
            return "respond"

        last_message = messages[-1].content
        span.set_attribute("node.input", last_message[:100])

        # Get last 5 messages for context to handle "yes", "ok" etc.
        recent_messages = messages[-5:]

        # Use LLM to classify intent
        response = await llm.ainvoke(
            [SystemMessage(content=ROUTER_SYSTEM_PROMPT)] + recent_messages
        )

        intent = response.content.strip().lower()
        span.set_attribute("node.output.intent", intent)

        logger.info(f"Routed intent: {intent}")

        if "reset" in intent:
            return "reset"
        elif "search" in intent:
            return "search_param"
        else:
            return "respond"
