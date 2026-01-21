# Copyright 2025 DataRobot, Inc.
# Licensed under the Apache License, Version 2.0
"""Status check node - looks up existing application status."""

import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

try:
    from state import AutoFinanceState
    from tools.supabase_client import check_application_status
except ImportError:
    from ..state import AutoFinanceState
    from ..tools.supabase_client import check_application_status


async def status_check_node(
    state: AutoFinanceState, config: RunnableConfig = None
) -> dict[str, Any]:
    """Look up status of existing application by Request ID."""
    messages = state.get("messages", [])

    # Get last user message
    last_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_msg = msg.content
            break

    # Look for request ID pattern
    match = re.search(r"af-\d{6,}-\d{4}", last_msg, re.IGNORECASE)

    if not match:
        return {
            "messages": [
                AIMessage(
                    content="Please provide your Request ID (format: AF-XXXXXX-XXXX)"
                )
            ],
            "current_phase": "onboarding",
        }

    request_id = match.group(0).upper()
    result = check_application_status.invoke(request_id, config=config)

    if result.get("found"):
        status_emoji = {
            "pending_review": "⏳",
            "approved": "✅",
            "rejected": "❌",
            "processing": "🔄",
        }.get(result.get("status", ""), "📋")

        return {
            "messages": [
                AIMessage(
                    content=f"""{status_emoji} **Application Status**

**Request ID:** {result.get("request_id")}
**Status:** {result.get("status", "Unknown").replace("_", " ").title()}
**Vehicle:** {result.get("vehicle_name", "N/A")}
**Submitted:** {result.get("created_at", "N/A")}

Need help with anything else?"""
                )
            ],
            "current_phase": "onboarding",
        }

    return {
        "messages": [
            AIMessage(
                content=f"❌ No application found: {request_id}\n\nPlease check the ID and try again."
            )
        ],
        "current_phase": "onboarding",
    }
