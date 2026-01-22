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
Question Nodes - Prompts for collecting financial profile data.

These nodes ask the user for specific information needed for loan assessment:
- Monthly income (EGP)
- Employment type
"""

from typing import Any

from langchain_core.messages import AIMessage
from models import AgentState, WorkflowPhase


def ask_income_node(state: AgentState) -> dict[str, Any]:
    """
    Ask the user for their monthly income.
    """
    # First, check if we need to ask for vehicle selection
    if not state.selected_vehicle and state.search_results:
        message = """
## 📋 Please Select a Vehicle

Before we proceed to financing, please select one of the vehicles from the search results.

Reply with the number of the vehicle you'd like to finance (e.g., "1" for the first option).
"""
        return {
            "messages": [AIMessage(content=message)],
            "current_phase": WorkflowPhase.PROFILING,
        }

    # If no search results yet, redirect to search
    if not state.selected_vehicle:
        message = """
## 🔍 No Vehicle Selected

It looks like we haven't found a vehicle yet. Please tell me what car you're looking for first.

For example: "I want a 2022 Hyundai Tucson under 1.5 million"
"""
        return {
            "messages": [AIMessage(content=message)],
            "current_phase": WorkflowPhase.DISCOVERY,
        }

    # Vehicle is selected, ask for income
    vehicle = state.selected_vehicle

    message = f"""
## 💰 Financial Profile - Step 1/2

Great choice! You've selected:

**{vehicle.name}** - {vehicle.price:,.0f} EGP

To calculate your monthly installment and check eligibility, I need some information.

**What is your monthly income in EGP?**

Please provide your approximate monthly income (e.g., "30000" or "30k EGP").
"""

    return {
        "messages": [AIMessage(content=message)],
        "current_phase": WorkflowPhase.PROFILING,
    }


def ask_employment_node(state: AgentState) -> dict[str, Any]:
    """
    Ask the user for their employment type.
    """
    income = state.monthly_income or 0

    message = f"""
## 💼 Financial Profile - Step 2/2

Thanks! I've recorded your monthly income as **{income:,.0f} EGP**.

**What is your employment type?**

Please choose one:
- **Corporate** - Employed at a corporate/large company
- **Salaried** - Regular employee with fixed salary
- **Self-Employed** - Freelancer, business owner, or independent worker
- **Other** - Retired, pensioner, or other income source

Just type your employment type (e.g., "Salaried" or "Corporate").
"""

    return {
        "messages": [AIMessage(content=message)],
        "current_phase": WorkflowPhase.PROFILING,
    }
