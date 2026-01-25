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
"""Debug tests for routing and confirmation flow."""

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from unittest.mock import AsyncMock, MagicMock

from agentic_workflow.nodes.router import route_intent, ROUTER_SYSTEM_PROMPT
from agentic_workflow.nodes.confirmation import check_confirmation, ConfirmationIntent
from agentic_workflow.models import AgentState, SearchParams


class TestRouterIntent:
    """Tests for router intent classification."""

    @pytest.mark.asyncio
    async def test_router_classifies_search_intent(self):
        """Test router correctly classifies search intent."""
        # Mock LLM that returns 'search'
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(content="search")

        state = {"messages": [HumanMessage(content="Find a 2024 Hyundai Tucson")]}

        result = await route_intent(state, mock_llm)

        assert result == "search_param"

    @pytest.mark.asyncio
    async def test_router_classifies_reset_intent(self):
        """Test router correctly classifies reset intent."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(content="reset")

        state = {"messages": [HumanMessage(content="start over")]}

        result = await route_intent(state, mock_llm)

        assert result == "reset"

    @pytest.mark.asyncio
    async def test_router_classifies_chat_intent(self):
        """Test router correctly classifies chat intent."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(content="chat")

        state = {"messages": [HumanMessage(content="Hello, what can you do?")]}

        result = await route_intent(state, mock_llm)

        assert result == "respond"


class TestConfirmationClassification:
    """Tests for confirmation LLM classification."""

    @pytest.mark.asyncio
    async def test_confirmation_classifies_yes_as_confirmed(self):
        """Test confirmation correctly classifies 'yes' as confirmed."""
        from agentic_workflow.nodes.confirmation import ConfirmationResult

        mock_llm = MagicMock()
        mock_structured = AsyncMock()
        mock_structured.ainvoke.return_value = ConfirmationResult(
            intent=ConfirmationIntent.CONFIRMED, reasoning="User said yes"
        )
        mock_llm.with_structured_output.return_value = mock_structured

        state = {"messages": [HumanMessage(content="yes")]}

        result = await check_confirmation(state, mock_llm)

        assert result["search_confirmed"] is True

    @pytest.mark.asyncio
    async def test_confirmation_classifies_no_as_cancelled(self):
        """Test confirmation correctly classifies 'no' as cancelled."""
        from agentic_workflow.nodes.confirmation import ConfirmationResult

        mock_llm = MagicMock()
        mock_structured = AsyncMock()
        mock_structured.ainvoke.return_value = ConfirmationResult(
            intent=ConfirmationIntent.CANCELLED, reasoning="User said no"
        )
        mock_llm.with_structured_output.return_value = mock_structured

        state = {"messages": [HumanMessage(content="no")]}

        result = await check_confirmation(state, mock_llm)

        assert result["search_confirmed"] is False
        assert result.get("search_params") is None  # Should clear params


class TestRouteDecision:
    """Tests for _route_decision logic in agent."""

    def test_route_decision_follows_router_when_no_params(self):
        """Test that route follows router when no search_params exist."""
        from agentic_workflow.agent import MyAgent

        agent = MyAgent()
        state = {
            "messages": [],
            "search_params": None,
            "search_confirmed": False,
            "_next_node": "search_param",
        }

        result = agent._route_decision(state)
        assert result == "search_param"

    def test_route_decision_goes_to_confirmation_when_params_pending(self):
        """Test that route goes to check_confirmation when params pending."""
        from agentic_workflow.agent import MyAgent

        agent = MyAgent()
        state = {
            "messages": [HumanMessage(content="yes")],
            "search_params": SearchParams(make="Toyota", model="Corolla"),
            "search_confirmed": False,
            "_next_node": "respond",
        }

        result = agent._route_decision(state)
        assert result == "check_confirmation"

    def test_route_decision_goes_to_search_when_confirmed(self):
        """Test that route goes to market_search when confirmed."""
        from agentic_workflow.agent import MyAgent

        agent = MyAgent()
        state = {
            "messages": [],
            "search_params": SearchParams(make="Toyota", model="Corolla"),
            "search_confirmed": True,
            "_next_node": "respond",
        }

        result = agent._route_decision(state)
        assert result == "market_search"


class TestGraphFlow:
    """Integration tests for graph flow."""

    def test_initial_search_request_goes_to_search_param(self):
        """Test that initial search request routes to search_param."""
        from agentic_workflow.agent import MyAgent

        agent = MyAgent()

        # Initial state - no search params
        state = {
            "messages": [HumanMessage(content="Find a Toyota Corolla")],
            "search_params": None,
            "search_confirmed": False,
            "_next_node": "search_param",  # Router should set this
        }

        result = agent._route_decision(state)

        # Should follow router's decision when no params
        assert result == "search_param"

    def test_confirmation_request_goes_to_check_confirmation(self):
        """Test that confirmation request routes to check_confirmation."""
        from agentic_workflow.agent import MyAgent

        agent = MyAgent()

        # State after search_param extracted params
        state = {
            "messages": [HumanMessage(content="yes please search")],
            "search_params": SearchParams(make="Toyota", model="Corolla"),
            "search_confirmed": False,
            "_next_node": "respond",  # Router might say chat, but we override
        }

        result = agent._route_decision(state)

        # Should go to check_confirmation regardless of router
        assert result == "check_confirmation"
