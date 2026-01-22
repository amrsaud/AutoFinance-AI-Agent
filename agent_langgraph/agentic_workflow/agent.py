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
AutoFinance AI Agent - Financial Co-Pilot for Vehicle Financing in Egypt.

This agent guides users through a 5-phase journey:
1. Onboarding & Routing
2. Market Discovery (Tavily Search)
3. Financial Profiling (Income, Employment)
4. Quotation (PMT Calculation)
5. Submission (Supabase Storage)
"""

from typing import Any, Literal

from config import Config
from datarobot_genai.langgraph.agent import LangGraphAgent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_litellm.chat_models import ChatLiteLLM
from langgraph.graph import END, START, StateGraph
from models import AgentState, WorkflowPhase
from nodes.ask_questions import ask_employment_node, ask_income_node
from nodes.market_search import market_search_node
from nodes.policy_rag import policy_rag_node
from nodes.profiling_logic import profiling_logic_node, route_profiling
from nodes.quotation import quotation_node
from nodes.router import router_node
from nodes.search_param import search_param_node
from nodes.status_check import status_check_node
from nodes.submission import submission_node

config = Config()


class MyAgent(LangGraphAgent):
    """
    AutoFinance AI Agent - A Financial Co-Pilot for vehicle financing in Egypt.

    This agent implements a stateful conversational workflow that:
    - Aggregates market data from Egyptian marketplaces (Hatla2ee, Dubizzle)
    - Enforces credit policies via RAG
    - Calculates monthly installments using PMT formula
    - Stores loan applications for back-office review

    The workflow uses a "Check-Ask-Exit" loop pattern where the agent
    halts execution until a new user message triggers resumption.
    """

    # Instance variables for memory persistence
    _checkpointer: Any = None
    _thread_id: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        """Initialize agent with optional checkpointer and thread_id for memory."""
        # Extract checkpointer and thread_id before passing to parent
        self._checkpointer = kwargs.pop("checkpointer", None)
        self._thread_id = kwargs.pop("thread_id", None)
        super().__init__(**kwargs)

    @property
    def langgraph_config(self) -> dict[str, Any]:
        """
        LangGraph configuration including thread_id for memory persistence.

        The thread_id in configurable enables the checkpointer to save and
        restore state across multiple invocations.
        """
        base_config: dict[str, Any] = {
            "recursion_limit": 150,
        }

        # Add thread_id for checkpointer if available
        if self._thread_id:
            base_config["configurable"] = {"thread_id": self._thread_id}

        return base_config

    async def _invoke(self, completion_create_params: Any) -> Any:
        """
        Override _invoke to compile graph with checkpointer for memory persistence.

        This overrides the base class to:
        1. Compile the workflow with the checkpointer (if available)
        2. Use thread_id in config for state persistence
        3. Smart input handling: only add new user message when resuming
        """
        from datarobot_genai.core.agents.base import (
            extract_user_prompt_content,
            is_streaming,
        )
        from langgraph.types import Command
        import logging

        logger = logging.getLogger(__name__)

        # Compile with checkpointer if available (enables memory)
        if self._checkpointer:
            langgraph_execution_graph = self.workflow.compile(
                checkpointer=self._checkpointer
            )
            logger.info(
                f"Graph compiled with checkpointer, thread_id: {self._thread_id}"
            )

            # Check if there's an existing checkpoint for this thread
            existing_checkpoint = None
            if self._thread_id:
                try:
                    existing_checkpoint = langgraph_execution_graph.get_state(
                        {"configurable": {"thread_id": self._thread_id}}
                    )
                except Exception as e:
                    logger.debug(f"No existing checkpoint found: {e}")

            if existing_checkpoint and existing_checkpoint.values:
                # Resume conversation - only add the new user message
                user_prompt = extract_user_prompt_content(completion_create_params)
                input_command = Command(
                    update={
                        "messages": [HumanMessage(content=user_prompt["input"])],
                    },
                )
                logger.info(
                    f"Resuming conversation, adding user message: {user_prompt['input'][:50]}..."
                )
            else:
                # New conversation - use full message list with system prompt
                input_command = self.convert_input_message(completion_create_params)
                logger.info("Starting new conversation with system prompt")
        else:
            langgraph_execution_graph = self.workflow.compile()
            input_command = self.convert_input_message(completion_create_params)
            logger.info("Graph compiled without checkpointer (no memory)")

        graph_stream = langgraph_execution_graph.astream(
            input=input_command,
            config=self.langgraph_config,
            debug=self.verbose,
            stream_mode=["updates", "messages"],
            subgraphs=True,
        )

        usage_metrics: dict[str, int] = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        }

        if is_streaming(completion_create_params):
            return self._stream_generator(graph_stream, usage_metrics)
        else:
            events: list[dict[str, Any]] = [
                event async for _, mode, event in graph_stream if mode == "updates"
            ]

            for update in events:
                current_node = next(iter(update))
                node_data = update[current_node]
                current_usage = (
                    node_data.get("usage", {}) if node_data is not None else {}
                )
                if current_usage:
                    usage_metrics["total_tokens"] += current_usage.get(
                        "total_tokens", 0
                    )
                    usage_metrics["prompt_tokens"] += current_usage.get(
                        "prompt_tokens", 0
                    )
                    usage_metrics["completion_tokens"] += current_usage.get(
                        "completion_tokens", 0
                    )

            pipeline_interactions = self.create_pipeline_interactions_from_events(
                events
            )

            last_event = events[-1]
            node_name = next(iter(last_event))
            node_data = last_event[node_name]
            response_text = (
                str(node_data["messages"][-1].content)
                if node_data is not None and "messages" in node_data
                else ""
            )

            return response_text, pipeline_interactions, usage_metrics

    @property
    def workflow(self) -> StateGraph[AgentState]:
        """
        Build the LangGraph StateGraph for the AutoFinance workflow.

        Graph Structure:
        START -> router -> [status_check | search_param | continue_flow]
        search_param -> [await confirmation]
        market_search -> profiling_logic
        profiling_logic -> [ask_income | ask_employment | policy_rag]
        policy_rag -> [quotation | rejection -> END]
        quotation -> [await confirmation]
        submission -> END
        """
        # Initialize the state graph with our custom state schema
        graph = StateGraph(AgentState)

        # Add all nodes
        graph.add_node("router", self._wrap_node(router_node))
        graph.add_node("search_param", self._wrap_node_with_llm(search_param_node))
        graph.add_node("market_search", self._wrap_node(market_search_node))
        graph.add_node("profiling_logic", self._wrap_node(profiling_logic_node))
        graph.add_node("ask_income", self._wrap_node(ask_income_node))
        graph.add_node("ask_employment", self._wrap_node(ask_employment_node))
        graph.add_node("policy_rag", self._wrap_node(policy_rag_node))
        graph.add_node("quotation", self._wrap_node(quotation_node))
        graph.add_node("submission", self._wrap_node(submission_node))
        graph.add_node("status_check", self._wrap_node(status_check_node))

        # Entry point - always start at router
        graph.add_edge(START, "router")

        # Router conditional edges
        graph.add_conditional_edges(
            "router",
            self._route_from_router,
            {
                "status_check": "status_check",
                "search_param": "search_param",
                "market_search": "market_search",
                "profiling_logic": "profiling_logic",
                "quotation": "quotation",
                "submission": "submission",
                END: END,
            },
        )

        # Search parameter extraction -> await user confirmation, then market search
        graph.add_edge("search_param", END)  # Pause for confirmation

        # Market search -> profiling
        graph.add_edge("market_search", "profiling_logic")

        # Profiling logic conditional edges (the data collection loop)
        graph.add_conditional_edges(
            "profiling_logic",
            route_profiling,
            {
                "ask_income": "ask_income",
                "ask_employment": "ask_employment",
                "policy_rag": "policy_rag",
            },
        )

        # Question nodes return to profiling for loop
        graph.add_edge("ask_income", END)  # Pause for user response
        graph.add_edge("ask_employment", END)  # Pause for user response

        # Policy RAG -> quotation (if eligible) or end (if rejected)
        graph.add_conditional_edges(
            "policy_rag",
            self._route_after_policy,
            {
                "quotation": "quotation",
                END: END,
            },
        )

        # Quotation -> pause for confirmation
        graph.add_edge("quotation", END)

        # Submission -> complete
        graph.add_edge("submission", END)

        # Status check -> complete
        graph.add_edge("status_check", END)

        # Return uncompiled graph - base class handles compilation
        return graph  # type: ignore[return-value]

    def _wrap_node(self, node_func):
        """Wrap a node function to handle state properly."""

        def wrapped(state: AgentState) -> dict[str, Any]:
            return node_func(state)

        return wrapped

    def _wrap_node_with_llm(self, node_func):
        """Wrap a node function that needs LLM access."""

        def wrapped(state: AgentState) -> dict[str, Any]:
            return node_func(state, llm=self.llm())

        return wrapped

    def _route_from_router(
        self, state: AgentState
    ) -> Literal[
        "status_check",
        "search_param",
        "market_search",
        "profiling_logic",
        "quotation",
        "submission",
        "__end__",
    ]:
        """
        Route from the router node based on current state and user input.

        This implements the main routing logic for the workflow.
        """
        # Get last message to determine intent
        last_message = state.messages[-1] if state.messages else None

        # Workflow complete
        if state.current_phase == WorkflowPhase.COMPLETED:
            return END

        # Check if this is a status check
        if isinstance(last_message, HumanMessage):
            user_text = str(last_message.content).lower()
            if "status" in user_text or "check" in user_text or "track" in user_text:
                import re

                uuid_pattern = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
                if re.search(uuid_pattern, user_text):
                    return "status_check"

        # Handle confirmations for search
        if state.awaiting_confirmation == "search":
            if isinstance(last_message, HumanMessage):
                user_text = str(last_message.content).lower().strip()
                if user_text in ["yes", "yeah", "yep", "ok", "okay", "confirm", "sure"]:
                    return "market_search"
                else:
                    return "search_param"  # Restart search

        # Handle confirmations for quote
        if state.awaiting_confirmation == "quote":
            if isinstance(last_message, HumanMessage):
                user_text = str(last_message.content).lower().strip()
                if user_text in [
                    "yes",
                    "yeah",
                    "yep",
                    "ok",
                    "okay",
                    "confirm",
                    "sure",
                    "proceed",
                ]:
                    return "submission"
                else:
                    return "search_param"  # Start over

        # Route based on current phase
        phase_routes = {
            WorkflowPhase.ONBOARDING: "search_param",
            WorkflowPhase.DISCOVERY: "search_param",
            WorkflowPhase.PROFILING: "profiling_logic",
            WorkflowPhase.QUOTATION: "quotation",
            WorkflowPhase.SUBMISSION: "submission",
        }

        return phase_routes.get(state.current_phase, "search_param")

    def _route_after_policy(self, state: AgentState) -> Literal["quotation", "__end__"]:
        """Route after policy RAG based on eligibility."""
        if state.applicable_policy and state.applicable_policy.is_eligible:
            return "quotation"
        return END  # Not eligible, end the flow

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        """System prompt template for the agent."""
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are AutoFinance AI, a Financial Co-Pilot helping users in Egypt find 
                and finance vehicles. You guide users through:
                1. Finding cars on Hatla2ee and Dubizzle
                2. Calculating affordable monthly installments
                3. Submitting pre-approval requests
                
                Always be helpful, professional, and transparent about loan terms.
                All prices are in Egyptian Pounds (EGP).
                """,
                ),
                (
                    "user",
                    "{input}",
                ),
            ]
        )

    def llm(
        self,
        preferred_model: str | None = None,
        auto_model_override: bool = True,
    ) -> ChatLiteLLM:
        """
        Returns the ChatLiteLLM to use for reasoning.

        Uses DataRobot LLM Gateway with temperature=0 for deterministic extraction.
        """
        api_base = self.litellm_api_base(config.llm_deployment_id)
        model = preferred_model
        if preferred_model is None:
            model = config.llm_default_model
        if auto_model_override and not config.use_datarobot_llm_gateway:
            model = config.llm_default_model
        if self.verbose:
            print(f"Using model: {model}")
        return ChatLiteLLM(
            model=model,
            api_base=api_base,
            api_key=self.api_key,
            timeout=self.timeout,
            streaming=True,
            max_retries=3,
            temperature=0.0,  # Deterministic for JSON extraction
        )
