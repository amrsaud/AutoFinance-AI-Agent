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
Submission Node - Collect PII and store application to Supabase.

This is the final step where we collect customer contact information
and persist the loan application for back-office review.
"""

import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from models import AgentState, CustomerInfo, WorkflowPhase
from tools.supabase_storage import create_application


def submission_node(state: AgentState) -> dict[str, Any]:
    """
    Collect customer information and submit application.

    If customer info is not yet collected, prompt for it.
    Otherwise, submit to Supabase and return confirmation.
    """
    # Check if we already have customer info
    if state.customer_info:
        # Submit the application
        return _submit_application(state)

    # Try to extract customer info from the last message
    customer_info = _extract_customer_info(state)

    if customer_info:
        # Store info and submit
        state_copy = state.model_copy()
        state_copy.customer_info = customer_info
        return _submit_application(state_copy)

    # Prompt for customer information
    message = """
## 📝 Almost There! Final Step

To complete your pre-approval request, please provide your contact information:

**Please share the following:**

1. **Full Name** - Your legal name as it appears on your ID
2. **Email Address** - For application updates
3. **Phone Number** - Egyptian mobile number
4. **National ID** (Optional) - For faster processing

You can provide this information in any format, for example:

> Name: Ahmed Mohamed
> Email: ahmed@example.com
> Phone: 01012345678
"""

    return {
        "messages": [AIMessage(content=message)],
        "current_phase": WorkflowPhase.SUBMISSION,
    }


def _extract_customer_info(state: AgentState) -> CustomerInfo | None:
    """
    Try to extract customer information from user messages.
    """
    if not state.messages:
        return None

    # Look for the most recent message with contact info
    last_message = None
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_message = msg
            break

    if not last_message:
        return None

    text = str(last_message.content)

    # Extract name
    name_patterns = [
        r"(?:name|اسم)[:\s]*([A-Za-z\u0600-\u06FF\s]{3,50})",
        r"^([A-Za-z\u0600-\u06FF\s]{3,50})(?:\n|$)",  # First line might be name
    ]
    name = None
    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            name = match.group(1).strip()
            break

    # Extract email
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    email = email_match.group(0) if email_match else None

    # Extract phone
    phone_patterns = [
        r"(?:phone|mobile|رقم|هاتف)[:\s]*(01[0-9]{9})",
        r"\b(01[0-9]{9})\b",  # Egyptian mobile format
        r"\b(\+20\s?1[0-9]{9})\b",  # International format
    ]
    phone = None
    for pattern in phone_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            phone = match.group(1).replace(" ", "")
            break

    # Extract National ID (14 digits)
    national_id_match = re.search(r"\b([0-9]{14})\b", text)
    national_id = national_id_match.group(1) if national_id_match else None

    # Require at least name, email, and phone
    if name and email and phone:
        return CustomerInfo(
            full_name=name,
            email=email,
            phone=phone,
            national_id=national_id,
        )

    return None


def _submit_application(state: AgentState) -> dict[str, Any]:
    """
    Submit the application to Supabase and return confirmation.
    """
    if (
        not state.customer_info
        or not state.selected_vehicle
        or not state.financial_quote
    ):
        return {
            "messages": [
                AIMessage(
                    content="⚠️ Missing required information. Please complete all steps."
                )
            ],
        }

    try:
        # Generate request ID and store application
        request_id = create_application(
            session_id=state.messages[0].id if state.messages else "unknown",
            customer_info=state.customer_info,
            vehicle=state.selected_vehicle,
            financial_quote=state.financial_quote,
            monthly_income=state.monthly_income or 0,
            employment_type=state.employment_type.value
            if state.employment_type
            else "other",
        )

        confirmation_message = f"""
## 🎉 Application Submitted Successfully!

### Your Request Details

| Item | Details |
|------|---------|
| **Request ID** | `{request_id}` |
| **Applicant** | {state.customer_info.full_name} |
| **Vehicle** | {state.selected_vehicle.name} |
| **Monthly Installment** | {state.financial_quote.monthly_installment:,.0f} EGP |
| **Status** | ⏳ Pending Review |

---

### What Happens Next?

1. 📧 You'll receive a confirmation email at **{state.customer_info.email}**
2. 📞 Our team will contact you at **{state.customer_info.phone}** within 24-48 hours
3. 📄 We may request additional documents for verification

---

### Save Your Request ID

**`{request_id}`**

Use this ID to check your application status anytime by saying:
> "Check status of {request_id}"

Thank you for choosing AutoFinance! 🚗
"""

        return {
            "messages": [AIMessage(content=confirmation_message)],
            "request_id": request_id,
            "current_phase": WorkflowPhase.COMPLETED,
        }

    except Exception as e:
        error_message = f"""
## ⚠️ Submission Error

There was an issue submitting your application:

```
{str(e)}
```

Don't worry - your information is saved. Please try again in a moment, 
or contact support if the issue persists.
"""
        return {
            "messages": [AIMessage(content=error_message)],
            "error_message": str(e),
        }
