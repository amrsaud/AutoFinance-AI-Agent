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
Reset node for clearing agent state.
Resets all state fields when user requests to start over.
"""

import logging

from langchain_core.messages import AIMessage
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


async def reset_state(state: dict) -> dict:
    """Reset the agent state to initial values.

    Clears search_params, search_results, and search_confirmed,
    and returns a confirmation message to the user.

    Args:
        state: The current agent state.

    Returns:
        dict: Updated state with cleared values and confirmation message.
    """
    with tracer.start_as_current_span("reset_state") as span:
        span.set_attribute("node.action", "state_reset")

        logger.info("Resetting agent state to initial values")

        # Return fresh state with confirmation message
        return {
            "search_params": None,
            "search_results": [],
            "search_confirmed": False,
            "messages": [
                AIMessage(
                    content="I've cleared everything and we're starting fresh! 🔄\n\n"
                    "How can I help you find a car today?"
                )
            ],
        }
