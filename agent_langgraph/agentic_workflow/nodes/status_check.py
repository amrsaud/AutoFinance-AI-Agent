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
Status Check Node - Query existing application status.

Allows users to check the status of a previously submitted
loan application using their Request ID.
"""

import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from models import AgentState
from tools.supabase_storage import format_status_response, get_application_status


def status_check_node(state: AgentState) -> dict[str, Any]:
    """
    Query and display application status.
    """
    # Extract request ID from user message
    request_id = _extract_request_id(state)

    if not request_id:
        message = """
## 🔍 Check Application Status

To check your application status, please provide your **Request ID**.

The Request ID is a unique code you received when submitting your application, 
formatted like: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`

For example: "Check status of 5379a74f-045e-4720-9c1b-3feb560d77ee"
"""
        return {
            "messages": [AIMessage(content=message)],
        }

    # Query Supabase for application status
    try:
        application = get_application_status(request_id)

        if application:
            status_message = format_status_response(application)
            return {
                "messages": [AIMessage(content=status_message)],
            }
        else:
            return {
                "messages": [
                    AIMessage(
                        content=f"""
## ❓ Application Not Found

I couldn't find an application with Request ID:
**`{request_id}`**

Please check that you've entered the correct ID. 

If you believe this is an error, please contact our support team.

---

Would you like to:
- **Try another ID** - Provide a different Request ID
- **Start new request** - Begin a new car financing application
"""
                    )
                ],
            }

    except Exception as e:
        return {
            "messages": [
                AIMessage(
                    content=f"""
## ⚠️ Error Checking Status

There was an issue querying your application:

```
{str(e)}
```

Please try again in a moment. If the issue persists, contact our support team.
"""
                )
            ],
            "error_message": str(e),
        }


def _extract_request_id(state: AgentState) -> str | None:
    """
    Extract UUID request ID from user messages.
    """
    if not state.messages:
        return None

    # Check the last few messages for a UUID
    uuid_pattern = (
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    )

    for msg in reversed(state.messages[-5:]):  # Check last 5 messages
        if isinstance(msg, HumanMessage):
            match = re.search(uuid_pattern, str(msg.content))
            if match:
                return match.group(0)

    return None
