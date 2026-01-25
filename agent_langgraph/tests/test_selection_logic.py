import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from models import Vehicle
from agentic_workflow.nodes.selection import selection_node


@pytest.mark.asyncio
async def test_selection_success():
    """Test successful selection from search results."""
    mock_llm = MagicMock()
    mock_out = MagicMock()
    mock_out.index = 1

    mock_runnable = MagicMock()
    mock_runnable.ainvoke = AsyncMock(return_value=mock_out)
    mock_llm.with_structured_output.return_value = mock_runnable

    vehicles = [
        Vehicle(
            make="Kia",
            model="Sportage",
            year=2024,
            price=1500000,
            source="x",
            source_url="u",
        )
    ]

    state = {
        "messages": [HumanMessage(content="I want option 1")],
        "search_results": vehicles,
    }

    result = await selection_node(state, mock_llm)

    assert "selected_vehicle" in result
    assert result["selected_vehicle"].model == "Sportage"
    assert "messages" in result


@pytest.mark.asyncio
async def test_selection_invalid_index():
    """Test invalid index selection."""
    mock_llm = MagicMock()
    mock_out = MagicMock()
    mock_out.index = 99  # Out of bounds

    mock_runnable = MagicMock()
    mock_runnable.ainvoke = AsyncMock(return_value=mock_out)
    mock_llm.with_structured_output.return_value = mock_runnable

    vehicles = [
        Vehicle(
            make="Kia",
            model="Sportage",
            year=2024,
            price=1500000,
            source="x",
            source_url="u",
        )
    ]

    state = {
        "messages": [HumanMessage(content="I want option 99")],
        "search_results": vehicles,
    }

    result = await selection_node(state, mock_llm)

    # Should not set selected_vehicle, should return message asking for clarification
    assert "selected_vehicle" not in result
    assert "messages" in result
    assert "couldn't identify" in result["messages"][0].content.lower()


@pytest.mark.asyncio
async def test_selection_no_results():
    """Test selection with no search results."""
    mock_llm = MagicMock()
    state = {"messages": [HumanMessage(content="Option 1")], "search_results": []}

    result = await selection_node(state, mock_llm)

    assert "selected_vehicle" not in result
    assert "Please search first" in result["messages"][0].content
