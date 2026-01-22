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
Market Search Node - Execute Tavily search and format results.

Searches Egyptian car marketplaces (Hatla2ee, Dubizzle) for vehicles
matching the user's search parameters.
"""

from typing import Any

from langchain_core.messages import AIMessage
from models import AgentState, WorkflowPhase
from tools.tavily_search import search_vehicles_sync


def market_search_node(state: AgentState) -> dict[str, Any]:
    """
    Execute marketplace search and present results to user.

    This node is triggered after the user confirms search parameters.
    """
    if not state.search_params:
        return {
            "messages": [
                AIMessage(
                    content="No search parameters found. Please tell me what car you're looking for."
                )
            ],
            "awaiting_confirmation": None,
        }

    params = state.search_params

    try:
        # Execute search
        vehicles = search_vehicles_sync(params, max_results=5)

        if not vehicles:
            return {
                "messages": [AIMessage(content=_no_results_message(params))],
                "search_results": [],
                "search_params_confirmed": True,
                "awaiting_confirmation": None,
                "current_phase": WorkflowPhase.DISCOVERY,
            }

        # Format results for display
        results_message = _format_search_results(vehicles)

        return {
            "messages": [AIMessage(content=results_message)],
            "search_results": vehicles,
            "search_params_confirmed": True,
            "awaiting_confirmation": None,
            "current_phase": WorkflowPhase.DISCOVERY,
        }

    except Exception as e:
        error_message = f"""
## ⚠️ Search Error

I encountered an issue while searching:

```
{str(e)}
```

Would you like to:
1. **Try again** - I'll retry the search
2. **Modify criteria** - Tell me different search parameters
"""
        return {
            "messages": [AIMessage(content=error_message)],
            "error_message": str(e),
            "awaiting_confirmation": None,
        }


def _no_results_message(params) -> str:
    """Generate message when no results found."""
    search_desc = []
    if params.make:
        search_desc.append(params.make)
    if params.model:
        search_desc.append(params.model)
    if params.year_min:
        search_desc.append(str(params.year_min))

    search_term = " ".join(search_desc) if search_desc else "your criteria"

    return f"""
## 🔍 No Results Found

I couldn't find any listings matching **{search_term}** on Egyptian marketplaces.

**Suggestions:**
- Try a different model year (older models may have more listings)
- Check for alternate spellings
- Remove the price cap if you set one
- Try a more popular model variant

What would you like to do?
"""


def _format_search_results(vehicles) -> str:
    """Format vehicle results for user display."""
    header = f"""
## 🚗 Found {len(vehicles)} Vehicles

Here are the best matches from Egyptian marketplaces:

"""

    results = []
    for i, vehicle in enumerate(vehicles, 1):
        mileage_str = f"{vehicle.mileage:,} km" if vehicle.mileage else "N/A"

        result = f"""
### {i}. {vehicle.name}

| Detail | Value |
|--------|-------|
| **Price** | {vehicle.price:,.0f} EGP |
| **Year** | {vehicle.year} |
| **Mileage** | {mileage_str} |
| **Source** | [{vehicle.source_site}]({vehicle.source_url}) |
"""
        results.append(result)

    footer = """
---

**To proceed with financing:**
Reply with the number of the vehicle you'd like to finance (e.g., "1" or "I want the first one").

Or tell me to **search again** with different criteria.
"""

    return header + "\n".join(results) + footer
