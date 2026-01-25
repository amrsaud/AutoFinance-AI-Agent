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
Respond node for conversational responses.
Generates conversational responses for general chat and follow-ups.
"""

import logging

from langchain_core.messages import SystemMessage
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

SYSTEM_PROMPT = """You are AutoFinance AI, a helpful assistant for finding and financing vehicles in Egypt.

This node is for GENERAL CONVERSATION ONLY. If you're receiving a message here, the user is:
- Asking general questions about your capabilities
- Saying hello or making small talk
- Following up on search results that were already executed
- Responding after you cancelled a search

Keep responses SHORT and helpful. Do NOT:
- Ask detailed questions about car preferences (that happens in search_param node)
- Pretend to search for cars (that happens in market_search node)
- Start a search conversation (redirect them to say what car they want)

If the user seems to want to search for a car, simply say:
"I can help you find a car! Just tell me what you're looking for (e.g., 'Find a 2024 Toyota Corolla')."

Remember: You are specialized in the Egyptian vehicle market (hatla2ee.com, dubizzle.com.eg)."""


async def respond(state: dict, llm) -> dict:
    """Generate conversational response.

    Args:
        state: The current agent state containing messages.
        llm: The language model to use for response generation.

    Returns:
        dict: Updated state with the assistant's response message.
    """
    with tracer.start_as_current_span("respond") as span:
        messages = state.get("messages", [])

        full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = await llm.ainvoke(full_messages)

        response_preview = response.content[:100] if response.content else ""
        span.set_attribute("node.output", response_preview)

        logger.info(f"Generated response: {response_preview}...")

        return {"messages": [response]}
