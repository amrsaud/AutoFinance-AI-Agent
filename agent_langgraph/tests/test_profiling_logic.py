import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from models import AgentState, UserProfile, EmploymentType
from agentic_workflow.nodes.profiling import profiling_node


@pytest.mark.asyncio
async def test_profiling_partial_update():
    """Test extracting partial info (Income) and asking for rest."""
    # Mock LLM (MagicMock for synchronous setup methods like with_structured_output)
    mock_llm = MagicMock()

    mock_extraction = MagicMock()
    mock_extraction.model_dump.return_value = {"monthly_income": 10000.0}

    # Mock structured output Runnable
    mock_runnable = MagicMock()
    mock_runnable.ainvoke = AsyncMock(return_value=mock_extraction)
    mock_llm.with_structured_output.return_value = mock_runnable

    state = {
        "messages": [HumanMessage(content="My income is 10000")],
        "user_profile": None,
    }

    result = await profiling_node(state, mock_llm)

    # Assert partial profile is saved
    assert "user_profile" in result
    assert result["user_profile"].monthly_income == 10000.0

    # Assert question is asked
    assert "messages" in result
    assert isinstance(result["messages"][0], AIMessage)


@pytest.mark.asyncio
async def test_profiling_complete():
    """Test complete profile transition."""
    # Mock LLM
    mock_llm = MagicMock()

    mock_extraction = MagicMock()
    mock_extraction.model_dump.return_value = {
        "employment_type": EmploymentType.FREELANCER_TECH,
        "contact_name": "John",
        "contact_phone": "123",
        "contact_email": "x@x.com",
        "existing_debt_obligations": 0.0,
    }

    mock_runnable = MagicMock()
    mock_runnable.ainvoke = AsyncMock(return_value=mock_extraction)
    mock_llm.with_structured_output.return_value = mock_runnable

    # State has income already
    current_profile = UserProfile(monthly_income=10000.0)
    state = {
        "messages": [HumanMessage(content="I am a freelancer named John")],
        "user_profile": current_profile,
    }

    result = await profiling_node(state, mock_llm)

    assert "user_profile" in result
    assert isinstance(result["user_profile"], UserProfile)
    assert result["user_profile"].contact_name == "John"
    # Result should include success message
    assert "messages" in result
