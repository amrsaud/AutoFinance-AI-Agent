# Copyright 2025 DataRobot, Inc.
# Licensed under the Apache License, Version 2.0
"""Selection node - uses interrupt() for human-in-the-loop vehicle selection."""

from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt, Command

try:
    from state import AutoFinanceState
except ImportError:
    from ..state import AutoFinanceState


async def selection_node(
    state: AutoFinanceState, config: RunnableConfig = None
) -> Command[Literal["profiling", "parse_search"]] | dict[str, Any]:
    """Pause for user to select a vehicle using interrupt().

    This node uses LangGraph's interrupt pattern to wait for human input.
    The interrupt payload includes vehicle options with source URLs for verification.
    """
    vehicles = state.get("search_results", [])

    if not vehicles:
        return {
            "messages": [
                AIMessage(content="No vehicles available. Please search again.")
            ],
            "current_phase": "onboarding",
        }

    # Build options with source URLs for user verification
    options = []
    for i, v in enumerate(vehicles, 1):
        # Handle both Pydantic and dict
        if hasattr(v, "model_dump"):
            vd = v.model_dump()
        elif isinstance(v, dict):
            vd = v
        else:
            vd = {
                "year": v.year,
                "make": v.make,
                "model": v.model,
                "price": v.price,
                "source_name": v.source_name,
                "source_url": v.source_url,
            }

        options.append(
            {
                "index": i,
                "label": f"{vd.get('year')} {vd.get('make')} {vd.get('model')} - {vd.get('price', 0):,.0f} EGP",
                "source": vd.get("source_name", "Unknown"),
                "url": vd.get("source_url", ""),
            }
        )

    # Pause execution - payload returned to caller under __interrupt__
    selected_idx = interrupt(
        {
            "type": "vehicle_selection",
            "options": options,
            "prompt": "Which vehicle would you like to finance? Check the source links to verify.",
            "allow_back": True,
        }
    )

    # Resume with user's choice
    if selected_idx is None or str(selected_idx).lower() in ["back", "cancel", "0"]:
        return Command(goto="parse_search", update={"current_phase": "parse_search"})

    try:
        idx = int(selected_idx) - 1
        if 0 <= idx < len(vehicles):
            selected = vehicles[idx]
            return Command(
                goto="profiling",
                update={
                    "selected_vehicle": selected,
                    "current_phase": "profiling",
                },
            )
    except (ValueError, TypeError):
        pass

    # Invalid selection - stay in selection
    return {
        "messages": [
            AIMessage(
                content=f"Please enter a number 1-{len(vehicles)} to select a vehicle."
            )
        ],
        "current_phase": "selection",
    }
