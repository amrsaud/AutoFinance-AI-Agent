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
"""Integration tests for complete graph flow debugging."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from agentic_workflow.models import AgentState, SearchParams
from agentic_workflow.agent import MyAgent


class TestGraphExecution:
    """Tests for actual graph execution flow."""

    @pytest.mark.asyncio
    async def test_first_search_request_flow(self):
        """Test that first search request goes through router -> search_param."""
        agent = MyAgent()
        workflow = agent.workflow

        # Compile without checkpointer for testing
        graph = workflow.compile()

        # Mock the LLM
        mock_llm_response = MagicMock()
        mock_llm_response.content = "search"

        mock_search_params = SearchParams(make="Hyundai", model="Tucson")

        with patch.object(agent, "llm") as mock_llm_method:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)
            mock_llm.with_structured_output = MagicMock(
                return_value=AsyncMock(
                    ainvoke=AsyncMock(return_value=mock_search_params)
                )
            )
            mock_llm_method.return_value = mock_llm

            # Execute graph with search request
            initial_state = {
                "messages": [HumanMessage(content="Find a 2024 Hyundai Tucson")]
            }

            # Collect all events
            events = []
            async for event in graph.astream(initial_state, debug=True):
                events.append(event)
                print(f"Event: {event}")

            # Get final state
            # Note: can't use aget_state without checkpointer, need to examine events
            print(f"Total events: {len(events)}")
            for i, e in enumerate(events):
                print(f"Event {i}: {e}")

    @pytest.mark.asyncio
    async def test_router_output_contains_next_node(self):
        """Test that router node returns _next_node."""
        agent = MyAgent()

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="search"))

        with patch.object(agent, "llm", return_value=mock_llm):
            state = AgentState(messages=[HumanMessage(content="Find a Toyota Corolla")])
            state_dict = state.model_dump()

            result = await agent._route_intent(state_dict)

            print(f"Router result: {result}")
            assert "_next_node" in result
            assert result["_next_node"] == "search_param"

    @pytest.mark.asyncio
    async def test_route_decision_with_fresh_state(self):
        """Test route decision when no search_params exist."""
        agent = MyAgent()

        # Fresh state - no search params
        state = {
            "messages": [HumanMessage(content="Find a Toyota Corolla")],
            "search_params": None,
            "search_confirmed": False,
            "_next_node": "search_param",
        }

        result = agent._route_decision(state)

        print(f"Route decision result: {result}")
        assert result == "search_param", f"Expected 'search_param', got '{result}'"

    @pytest.mark.asyncio
    async def test_route_decision_with_pending_params(self):
        """Test route decision when search_params exist but not confirmed."""
        agent = MyAgent()

        # State with pending search params
        state = {
            "messages": [HumanMessage(content="yes")],
            "search_params": SearchParams(make="Toyota", model="Corolla"),
            "search_confirmed": False,
            "_next_node": "respond",  # Router might say chat
        }

        result = agent._route_decision(state)

        print(f"Route decision result: {result}")
        assert result == "check_confirmation", (
            f"Expected 'check_confirmation', got '{result}'"
        )

    @pytest.mark.asyncio
    async def test_search_param_returns_params_and_message(self):
        """Test that search_param node returns search_params and confirmation message."""
        agent = MyAgent()

        mock_search_params = SearchParams(make="Hyundai", model="Tucson", year_min=2024)

        mock_llm = MagicMock()
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_search_params)
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured)

        with patch.object(agent, "llm", return_value=mock_llm):
            state = {"messages": [HumanMessage(content="Find a 2024 Hyundai Tucson")]}

            result = await agent._extract_search_params(state)

            print(f"Search param result: {result}")
            assert "search_params" in result, "Expected 'search_params' in result"
            assert result["search_params"] is not None, (
                "search_params should not be None"
            )
            assert "messages" in result, "Expected 'messages' in result"
            assert len(result["messages"]) > 0, "Should have confirmation message"
            print(f"Confirmation message: {result['messages'][0].content}")


class TestStateTransitions:
    """Test state transitions between nodes."""

    def test_search_params_model_serialization(self):
        """Test that SearchParams can be serialized to state dict."""
        params = SearchParams(make="Toyota", model="Corolla", year_min=2024)
        params_dict = params.model_dump()

        print(f"Serialized params: {params_dict}")
        assert params_dict["make"] == "Toyota"
        assert params_dict["model"] == "Corolla"

    def test_agent_state_includes_search_fields(self):
        """Test AgentState has all required fields."""
        state = AgentState(
            messages=[], search_params=SearchParams(make="BMW"), search_confirmed=False
        )

        print(f"State fields: {state.model_fields.keys()}")
        assert hasattr(state, "search_params")
        assert hasattr(state, "search_confirmed")
        assert hasattr(state, "search_results")
        assert hasattr(state, "messages")


if __name__ == "__main__":
    import asyncio

    # Run quick debug
    async def debug():
        agent = MyAgent()

        # Test route decision
        print("\n=== Testing route_decision ===")
        state_fresh = {
            "messages": [HumanMessage(content="Find a car")],
            "search_params": None,
            "_next_node": "search_param",
        }
        print(f"Fresh state decision: {agent._route_decision(state_fresh)}")

        state_pending = {
            "messages": [HumanMessage(content="yes")],
            "search_params": SearchParams(make="Toyota"),
            "search_confirmed": False,
            "_next_node": "respond",
        }
        print(f"Pending state decision: {agent._route_decision(state_pending)}")

    asyncio.run(debug())
