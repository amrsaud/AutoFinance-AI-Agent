import logging
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from opentelemetry import trace
from pydantic import BaseModel, Field

from models import Vehicle

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

SELECTION_SYSTEM_PROMPT = """You are a helper extracting the user's vehicle selection.
The user will select a vehicle from a numbered list (e.g., "Option 1", "The first one", "The BMW").
You have a list of available vehicles.
Your goal is to identify functionality which index (1-based) the user selected.

Return the 1-based index of the selected vehicle.
If you cannot determine the selection, return 0.
"""


class SelectionOutput(BaseModel):
    """Extracted selection index."""

    index: int = Field(
        ..., description="1-based index of the selected vehicle. 0 if not found."
    )


async def selection_node(state: dict, llm) -> dict:
    """Parses user selection and updates selected_vehicle state.

    Args:
        state: Current graph state.
        llm: Language model.

    Returns:
        dict: Update to state (selected_vehicle, messages).
    """
    with tracer.start_as_current_span("selection_node") as span:
        messages = state.get("messages", [])
        search_results = state.get("search_results", [])

        if not search_results:
            return {
                "messages": [
                    AIMessage(
                        content="I don't have a list of vehicles to select from. Please search first."
                    )
                ]
            }

        if messages and isinstance(messages[-1], HumanMessage):
            last_msg = messages[-1].content

            # Context for LLM
            # We don't need to dump ALL vehicles, just say "User sees a list of X vehicles".
            # Or we can provide a simplified list for fuzzy matching if we want.
            # For now, let's try direct index extraction or fuzzy match by name.

            # Simple Regex check for "Option X" or just "X"?
            # Let's use LLM for robustness.

            structured_llm = llm.with_structured_output(SelectionOutput)
            result = await structured_llm.ainvoke(
                [
                    SystemMessage(content=SELECTION_SYSTEM_PROMPT),
                    HumanMessage(
                        content=f"Context: {len(search_results)} vehicles available.\nUser Input: {last_msg}"
                    ),
                ]
            )

            idx = result.index

            if 1 <= idx <= len(search_results):
                vehicle = search_results[idx - 1]
                logger.info(
                    f"User selected vehicle index {idx}: {vehicle.make} {vehicle.model}"
                )

                return {
                    "selected_vehicle": vehicle,
                    "messages": [
                        AIMessage(
                            content=f"Great choice! I've selected the **{vehicle.make} {vehicle.model}** ({vehicle.price:,} EGP). \n\nTo check financing, could you please tell me your **monthly income**?"
                        )
                    ],
                }
            else:
                return {
                    "messages": [
                        AIMessage(
                            content="I couldn't identify which vehicle you selected. Please specify the number (e.g., 'Option 1')."
                        )
                    ]
                }

        return {"messages": [AIMessage(content="Please select a vehicle number.")]}
