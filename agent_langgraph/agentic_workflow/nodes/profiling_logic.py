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
Profiling Logic Node - Pure logic for data collection routing.

This is a logic-only node (no LLM) that checks the state for
required financial profile data and routes to appropriate
collection nodes.
"""

import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from models import AgentState, EmploymentType, WorkflowPhase


def profiling_logic_node(state: AgentState) -> dict[str, Any]:
    """
    Check if required financial data is collected.

    This node routes based on what data is still needed.
    Updates current_phase to PROFILING if not already.
    """
    # If vehicle not selected yet, try to extract from last message
    if not state.selected_vehicle and state.search_results:
        selected = _extract_vehicle_selection(state)
        if selected:
            return {
                "selected_vehicle": selected,
                "current_phase": WorkflowPhase.PROFILING,
            }

    # Try to extract income/employment from the last message if provided
    updates = _extract_financial_data(state)
    if updates:
        return updates

    # No changes, just update phase
    return {
        "current_phase": WorkflowPhase.PROFILING,
    }


def route_profiling(
    state: AgentState,
) -> Literal["ask_income", "ask_employment", "policy_rag"]:
    """
    Determine which node to route to based on collected data.

    This implements the cyclic data collection loop from the technical design.
    """
    # Check if vehicle is selected
    if not state.selected_vehicle:
        return "ask_income"  # Will prompt to select vehicle first

    # Check required fields in order
    if state.monthly_income is None:
        return "ask_income"

    if state.employment_type is None:
        return "ask_employment"

    # All data collected, proceed to policy check
    return "policy_rag"


def _extract_vehicle_selection(state: AgentState) -> Any | None:
    """
    Try to extract vehicle selection from user message.

    Handles patterns like:
    - "1" or "2" (number selection)
    - "first one" or "second one"
    - "I want the Hyundai"
    """
    if not state.messages:
        return None

    last_message = None
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_message = msg
            break

    if not last_message:
        return None

    user_text = str(last_message.content).lower().strip()
    vehicles = state.search_results

    if not vehicles:
        return None

    # Pattern 1: Direct number (1-5)
    if user_text.isdigit():
        idx = int(user_text) - 1
        if 0 <= idx < len(vehicles):
            return vehicles[idx]

    # Pattern 2: Ordinal words
    ordinals = {
        "first": 0,
        "second": 1,
        "third": 2,
        "fourth": 3,
        "fifth": 4,
        "1st": 0,
        "2nd": 1,
        "3rd": 2,
        "4th": 3,
        "5th": 4,
    }
    for word, idx in ordinals.items():
        if word in user_text and idx < len(vehicles):
            return vehicles[idx]

    # Pattern 3: Number in text
    match = re.search(r"\b([1-5])\b", user_text)
    if match:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(vehicles):
            return vehicles[idx]

    return None


def _extract_financial_data(state: AgentState) -> dict[str, Any] | None:
    """
    Try to extract income and employment data from user messages.
    """
    if not state.messages:
        return None

    last_message = None
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_message = msg
            break

    if not last_message:
        return None

    user_text = str(last_message.content)
    updates = {}

    # Extract income if not yet collected
    if state.monthly_income is None:
        income = _extract_income(user_text)
        if income:
            updates["monthly_income"] = income

    # Extract employment type if not yet collected
    if state.employment_type is None:
        employment = _extract_employment_type(user_text)
        if employment:
            updates["employment_type"] = employment

    return updates if updates else None


def _extract_income(text: str) -> float | None:
    """Extract monthly income from text."""
    text_lower = text.lower()

    # Pattern: number followed by currency indicator
    patterns = [
        r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:k|thousand|ألف)",  # 30k, 30 thousand
        r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:egp|جنيه|pounds?|le)",
        r"(?:income|salary|راتب).*?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)",
        r"(\d{4,})",  # Plain number > 1000
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower.replace(",", ""))
        if match:
            value = float(match.group(1))
            # Handle "k" multiplier
            if "k" in text_lower or "thousand" in text_lower or "ألف" in text_lower:
                value *= 1000
            # Reasonable income range check (1000 - 500000 EGP)
            if 1000 <= value <= 500000:
                return value

    return None


def _extract_employment_type(text: str) -> EmploymentType | None:
    """Extract employment type from text."""
    text_lower = text.lower()

    patterns = {
        EmploymentType.CORPORATE: [
            "corporate",
            "company",
            "corporate employee",
            "شركة",
        ],
        EmploymentType.SALARIED: ["salaried", "employee", "موظف", "salary", "wage"],
        EmploymentType.SELF_EMPLOYED: [
            "self-employed",
            "self employed",
            "freelance",
            "business owner",
            "حر",
            "عمل حر",
        ],
        EmploymentType.OTHER: ["other", "retired", "pension", "متقاعد", "أخرى"],
    }

    for emp_type, keywords in patterns.items():
        for keyword in keywords:
            if keyword in text_lower:
                return emp_type

    return None
