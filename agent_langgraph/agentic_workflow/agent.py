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
AutoFinance AI Agent - Modular LangGraph Architecture.

This agent uses a modular node structure with:
- create_react_agent for LLM-driven nodes (welcome, parse_search, execute_search)
- interrupt() for human-in-the-loop (selection, quotation)
- Custom functions for deterministic logic (validation, profiling, lead_capture)
"""

from typing import Any, Literal

from datarobot_genai.langgraph.agent import LangGraphAgent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_litellm.chat_models import ChatLiteLLM
from langgraph.graph import END, START, StateGraph

try:
    from config import Config
    from state import AutoFinanceState
    from nodes.welcome import create_welcome_node
    from nodes.parse_search import parse_search_node
    from nodes.validation import validation_node
    from nodes.execute_search import execute_search_node
    from nodes.selection import selection_node
    from nodes.profiling import profiling_node
    from nodes.quotation import quotation_node
    from nodes.lead_capture import lead_capture_node
    from nodes.status_check import status_check_node
except ImportError:
    from .config import Config
    from .state import AutoFinanceState
    from .nodes.welcome import create_welcome_node
    from .nodes.parse_search import parse_search_node
    from .nodes.validation import validation_node
    from .nodes.execute_search import execute_search_node
    from .nodes.selection import selection_node
    from .nodes.profiling import profiling_node
    from .nodes.quotation import quotation_node
    from .nodes.lead_capture import lead_capture_node
    from .nodes.status_check import status_check_node

config = Config()


class AutoFinanceAgent(LangGraphAgent):
    """AutoFinance AI Agent with modular LangGraph architecture."""

    @property
    def workflow(self) -> StateGraph[AutoFinanceState]:
        """Build modular workflow with proper nodes and edges."""
        wf = StateGraph(AutoFinanceState)

        # Add nodes
        wf.add_node("welcome", self._welcome_node)
        wf.add_node("parse_search", self._parse_search_node)
        wf.add_node("validation", self._validation_node)
        wf.add_node("execute_search", self._execute_search_node)
        wf.add_node("selection", self._selection_node)
        wf.add_node("profiling", self._profiling_node)
        wf.add_node("quotation", self._quotation_node)
        wf.add_node("lead_capture", self._lead_capture_node)
        wf.add_node("status_check", self._status_check_node)

        # Entry point
        wf.add_edge(START, "welcome")

        # Conditional edges from welcome
        wf.add_conditional_edges(
            "welcome",
            self._route_from_welcome,
            {
                "parse_search": "parse_search",
                "status_check": "status_check",
                "end": END,
            },
        )

        # Parse search -> validation
        wf.add_edge("parse_search", "validation")

        # Conditional edges from validation
        wf.add_conditional_edges(
            "validation",
            self._route_from_validation,
            {
                "execute_search": "execute_search",
                "parse_search": "parse_search",
                "end": END,
            },
        )

        # Execute search -> selection
        wf.add_edge("execute_search", "selection")

        # Selection uses Command to route (interrupt pattern)
        # After interrupt resume, it routes via Command(goto=...)

        # Profiling conditional edges
        wf.add_conditional_edges(
            "profiling",
            self._route_from_profiling,
            {
                "quotation": "quotation",
                "end": END,
            },
        )

        # Quotation uses Command to route (interrupt pattern)

        # Lead capture -> END
        wf.add_edge("lead_capture", END)

        # Status check -> END
        wf.add_edge("status_check", END)

        return wf  # type: ignore

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([("user", "{input}")])

    def llm(
        self, preferred_model: str | None = None, auto_model_override: bool = True
    ) -> ChatLiteLLM:
        api_base = self.litellm_api_base(config.llm_deployment_id)
        model = preferred_model or config.llm_default_model
        if auto_model_override and not config.use_datarobot_llm_gateway:
            model = config.llm_default_model
        return ChatLiteLLM(
            model=model,
            api_base=api_base,
            api_key=self.api_key,
            timeout=self.timeout,
            streaming=True,
            max_retries=3,
        )

    # ========== Node Wrappers ==========

    async def _welcome_node(
        self, state: AutoFinanceState, config: RunnableConfig = None
    ) -> dict[str, Any]:
        """Welcome node using create_react_agent."""
        agent = create_welcome_node(self.llm())
        return await agent.ainvoke(state, config=config)

    async def _parse_search_node(
        self, state: AutoFinanceState, config: RunnableConfig = None
    ) -> dict[str, Any]:
        """Parse search node wrapper."""
        return await parse_search_node(state, config, llm=self.llm())

    async def _validation_node(
        self, state: AutoFinanceState, config: RunnableConfig = None
    ) -> dict[str, Any]:
        """Validation node wrapper."""
        return await validation_node(state, config)

    async def _execute_search_node(
        self, state: AutoFinanceState, config: RunnableConfig = None
    ) -> dict[str, Any]:
        """Execute search node wrapper."""
        return await execute_search_node(state, config, llm=self.llm())

    async def _selection_node(
        self, state: AutoFinanceState, config: RunnableConfig = None
    ) -> dict[str, Any]:
        """Selection node with interrupt() wrapper."""
        return await selection_node(state, config)

    async def _profiling_node(
        self, state: AutoFinanceState, config: RunnableConfig = None
    ) -> dict[str, Any]:
        """Profiling node wrapper."""
        return await profiling_node(state, config)

    async def _quotation_node(
        self, state: AutoFinanceState, config: RunnableConfig = None
    ) -> dict[str, Any]:
        """Quotation node with interrupt() wrapper."""
        return await quotation_node(state, config)

    async def _lead_capture_node(
        self, state: AutoFinanceState, config: RunnableConfig = None
    ) -> dict[str, Any]:
        """Lead capture node wrapper."""
        return await lead_capture_node(state, config)

    async def _status_check_node(
        self, state: AutoFinanceState, config: RunnableConfig = None
    ) -> dict[str, Any]:
        """Status check node wrapper."""
        return await status_check_node(state, config)

    # ========== Routing Functions ==========

    def _route_from_welcome(
        self, state: AutoFinanceState
    ) -> Literal["parse_search", "status_check", "end"]:
        """Route from welcome node based on user intent."""
        messages = state.get("messages", [])

        # Get last user message
        last_msg = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_msg = msg.content.lower()
                break

        # Check for status request
        if any(w in last_msg for w in ["status", "check", "af-", "طلب", "حالة"]):
            return "status_check"

        # Check for search intent
        if any(
            w in last_msg
            for w in [
                "find",
                "search",
                "looking",
                "want",
                "need",
                "سيارة",
                "ابحث",
                "عايز",
            ]
        ):
            return "parse_search"

        # Default to parse_search for any car-related query
        return "parse_search"

    def _route_from_validation(
        self, state: AutoFinanceState
    ) -> Literal["execute_search", "parse_search", "end"]:
        """Route from validation based on user confirmation."""
        result = state.get("validation_result", "")

        if result == "confirmed":
            return "execute_search"
        elif result == "modify":
            return "parse_search"
        else:
            return "end"

    def _route_from_profiling(
        self, state: AutoFinanceState
    ) -> Literal["quotation", "end"]:
        """Route from profiling based on eligibility."""
        phase = state.get("current_phase", "")

        if phase == "quotation":
            return "quotation"
        else:
            return "end"


MyAgent = AutoFinanceAgent
