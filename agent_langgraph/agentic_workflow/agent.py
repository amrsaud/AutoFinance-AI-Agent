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
AutoFinance AI Agent - Market Discovery Agent with LangGraph.

A conversational agent that helps users find vehicles in Egypt
by searching hatla2ee.com and dubizzle.com.eg.
"""

import uuid
from typing import Any

from config import Config
from datarobot_genai.langgraph.agent import LangGraphAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain_litellm.chat_models import ChatLiteLLM
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from models import AgentState
from nodes import (
    check_confirmation,
    extract_search_params,
    reset_state,
    respond,
    route_intent,
    search_market,
    should_execute_search,
)

config = Config()


class MyAgent(LangGraphAgent):
    """AutoFinance AI Agent for vehicle market discovery in Egypt.

    Uses LangGraph checkpointing with SQLite for persistent memory
    across conversations. Supports:
    - Vehicle search via Tavily API (hatla2ee.com, dubizzle.com.eg)
    - Human-in-the-loop confirmation before search
    - State reset on user request

    Graph Flow:
    1. User: "Find a Hyundai Tucson" → router → search_param → END (ask confirmation)
    2. User: "yes" → router → check_confirmation → market_search → respond → END
    """

    def __init__(
        self,
        **kwargs: Any,
    ):
        """Initialize the agent.

        Args:
            **kwargs: Additional arguments passed to LangGraphAgent.
        """
        super().__init__(**kwargs)

    @property
    def checkpointer(self) -> None:
        """Deprecated: Checkpointer is managed in _invoke."""
        return None

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        """Prompt template for user input processing."""
        return ChatPromptTemplate.from_messages(
            [
                ("user", "{user_prompt}"),
            ]
        )

    @property
    def workflow(self) -> StateGraph:
        """Build the Market Discovery agent workflow.

        Graph Structure:
        - router: Classifies intent (search/confirm/reset/chat)
        - search_param: Extracts search parameters, asks for confirmation
        - check_confirmation: Validates user confirmation
        - market_search: Executes Tavily search
        - respond: Generates conversational responses
        - reset: Resets state when user requests

        Human-in-the-Loop Flow:
        1. search_param extracts params → END (pauses for user confirmation)
        2. User confirms → router detects confirmation → check_confirmation
        3. If confirmed → market_search → respond → END

        Returns:
            StateGraph: The configured workflow graph.
        """
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("router", self._route_intent)
        workflow.add_node("search_param", self._extract_search_params)
        workflow.add_node("check_confirmation", self._check_confirmation)
        workflow.add_node("market_search", self._search_market)
        workflow.add_node("respond", self._respond)
        workflow.add_node("reset", self._reset_state)

        # Add edges
        workflow.add_edge(START, "router")

        # Router decides where to go based on intent and state
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "search_param": "search_param",
                "check_confirmation": "check_confirmation",
                "market_search": "market_search",  # Direct route if already confirmed
                "respond": "respond",
                "reset": "reset",
            },
        )

        # search_param ends (pause for confirmation)
        workflow.add_edge("search_param", END)

        # check_confirmation decides whether to search or end
        workflow.add_conditional_edges(
            "check_confirmation",
            self._should_execute_search,
            {
                "market_search": "market_search",
                "end": END,
            },
        )

        workflow.add_edge("market_search", "respond")
        workflow.add_edge("respond", END)
        workflow.add_edge("reset", END)

        return workflow

    def _route_decision(self, state: AgentState) -> str:
        """Determine the next node based on intent and state.

        Routing Logic:
        1. If we have pending search_params (not confirmed) → check_confirmation (LLM classifies)
        2. If already confirmed and search_params exist → market_search
        3. Otherwise → follow router's decision (search_param/reset/respond)

        The LLM classification happens in check_confirmation node, not here.
        """
        search_params = state.get("search_params")
        search_confirmed = state.get("search_confirmed", False)
        next_node_from_router = state.get("_next_node", "respond")

        # If router detected RESET, prioritize that even if searching
        if next_node_from_router == "reset":
            return "reset"

        # If we have pending search params (waiting for confirmation)
        # Route to check_confirmation where LLM will classify the intent
        if search_params and not search_confirmed:
            return "check_confirmation"

        # If already confirmed and search_params exist, can go directly to search
        if search_params and search_confirmed:
            return "market_search"

        # Follow router's decision
        return next_node_from_router

    async def _route_intent(self, state: AgentState) -> dict:
        """Route user intent to appropriate node."""
        llm = self.llm()
        next_node = await route_intent(state, llm)
        return {"_next_node": next_node}

    async def _extract_search_params(self, state: AgentState) -> dict:
        """Extract search parameters from user query."""
        llm = self.llm()
        return await extract_search_params(state, llm)

    async def _check_confirmation(self, state: AgentState) -> dict:
        """Check if user confirmed the search using LLM classification."""
        llm = self.llm()
        return await check_confirmation(state, llm)

    def _should_execute_search(self, state: AgentState) -> str:
        """Determine if search should be executed."""
        return should_execute_search(state)

    async def _search_market(self, state: AgentState) -> dict:
        """Execute market search via Tavily."""
        llm = self.llm()
        return await search_market(state, llm)

    async def _respond(self, state: AgentState) -> dict:
        """Generate conversational response."""
        llm = self.llm()
        return await respond(state, llm)

    async def _reset_state(self, state: AgentState) -> dict:
        """Reset agent state to initial values."""
        return await reset_state(state)

    async def _invoke(self, completion_create_params):
        """Override to compile workflow with checkpointer for memory persistence."""
        from datarobot_genai.langgraph.agent import is_streaming

        input_command = self.convert_input_message(completion_create_params)
        run_config = self._get_run_config(completion_create_params)

        if config.sqlite_path:
            if is_streaming(completion_create_params):
                # We need the context manager to be active DURING the iteration
                async def wrapped_generator():
                    async with AsyncSqliteSaver.from_conn_string(
                        config.sqlite_path
                    ) as checkpointer:
                        agent_graph = self.workflow.compile(checkpointer=checkpointer)
                        async for item in self._execute_graph_stream(
                            input_command,
                            completion_create_params,
                            agent_graph,
                            run_config,
                        ):
                            yield item

                return wrapped_generator()
            else:
                async with AsyncSqliteSaver.from_conn_string(
                    config.sqlite_path
                ) as checkpointer:
                    agent_graph = self.workflow.compile(checkpointer=checkpointer)
                    return await self._execute_graph_sync(
                        input_command, completion_create_params, agent_graph, run_config
                    )
        else:
            agent_graph = self.workflow.compile()
            if is_streaming(completion_create_params):
                return self._execute_graph_stream(
                    input_command, completion_create_params, agent_graph, run_config
                )
            else:
                return await self._execute_graph_sync(
                    input_command, completion_create_params, agent_graph, run_config
                )

    def _get_run_config(self, completion_create_params) -> dict:
        """Extract thread_id and build run config."""
        extra_body = completion_create_params.get("extra_body") or {}
        metadata = completion_create_params.get("metadata") or {}
        extra_body_metadata = extra_body.get("metadata") or {}

        request_thread_id = (
            extra_body.get("thread_id")
            or extra_body.get("datarobot_association_id")
            or extra_body.get("association_id")
            or extra_body.get("chatId")
            or extra_body_metadata.get("thread_id")
            or completion_create_params.get("thread_id")
            or completion_create_params.get("datarobot_association_id")
            or completion_create_params.get("association_id")
            or completion_create_params.get("chatId")
            or metadata.get("thread_id")
            or str(uuid.uuid4())
        )
        return {"configurable": {"thread_id": request_thread_id}}

    async def _execute_graph_stream(
        self, input_command, completion_create_params, compiled_graph, run_config
    ):
        """Execute graph and stream results correctly."""
        # Unwrap Command if present to ensure state update triggers START
        graph_input = input_command
        if hasattr(input_command, "update"):
            graph_input = input_command.update

        graph_stream = compiled_graph.astream(
            input=graph_input,
            config=run_config,
            debug=self.verbose,
            stream_mode=["messages", "updates"],
            subgraphs=True,
        )

        usage_metrics = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        }

        # Use our custom generator that handles AIMessage
        async for item in self._stream_generator(graph_stream, usage_metrics):
            yield item

    async def _execute_graph_sync(
        self, input_command, completion_create_params, compiled_graph, run_config
    ):
        """Execute graph synchronously."""
        # Unwrap Command if present
        graph_input = input_command
        if hasattr(input_command, "update"):
            graph_input = input_command.update

        # Use parent implementation logic for sync execution or simplified one
        # Calling parent logic is tricky without super()._invoke, so we replicate the sync logic
        # but using astream for consistency

        graph_stream = compiled_graph.astream(
            input=graph_input,
            config=run_config,
            debug=self.verbose,
            stream_mode=["updates"],
            subgraphs=True,
        )

        events = []
        usage_metrics = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        }

        async for _, mode, event in graph_stream:
            if mode == "updates":
                events.append(event)
                # Accumulate metrics
                current_node = next(iter(event))
                node_data = event[current_node]
                current_usage = node_data.get("usage", {}) if node_data else {}
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

        pipeline_interactions = self.create_pipeline_interactions_from_events(events)

        state = await compiled_graph.aget_state(run_config)
        messages = state.values.get("messages", [])
        response_text = str(messages[-1].content) if messages else ""

        return response_text, pipeline_interactions, usage_metrics

    async def _stream_generator(self, graph_stream, usage_metrics):
        """Override stream generator to handle AIMessage objects."""
        from ag_ui.core import (
            EventType,
            TextMessageContentEvent,
            TextMessageEndEvent,
            TextMessageStartEvent,
            ToolCallArgsEvent,
            ToolCallEndEvent,
            ToolCallResultEvent,
            ToolCallStartEvent,
        )
        from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

        current_message_id = None
        tool_call_id = ""

        async for _, mode, event in graph_stream:
            if mode == "messages":
                message_event = event
                message = message_event[0]

                if isinstance(message, ToolMessage):
                    # Handle ToolMessage (Original Logic)
                    yield (
                        ToolCallEndEvent(
                            type=EventType.TOOL_CALL_END,
                            tool_call_id=message.tool_call_id,
                        ),
                        None,
                        usage_metrics,
                    )
                    yield (
                        ToolCallResultEvent(
                            type=EventType.TOOL_CALL_RESULT,
                            message_id=message.id,
                            tool_call_id=message.tool_call_id,
                            content=message.content,
                            role="tool",
                        ),
                        None,
                        usage_metrics,
                    )
                    tool_call_id = ""

                elif isinstance(message, (AIMessageChunk, AIMessage)):
                    # Handle AIMessageChunk AND AIMessage
                    if (
                        hasattr(message, "tool_call_chunks")
                        and message.tool_call_chunks
                    ):
                        # Logic for tool calls (chunked)
                        for tool_call_chunk in message.tool_call_chunks:
                            if name := tool_call_chunk.get("name"):
                                tool_call_id = tool_call_chunk["id"]
                                yield (
                                    ToolCallStartEvent(
                                        type=EventType.TOOL_CALL_START,
                                        tool_call_id=tool_call_id,
                                        tool_call_name=name,
                                        parent_message_id=message.id,
                                    ),
                                    None,
                                    usage_metrics,
                                )
                            elif args := tool_call_chunk.get("args"):
                                yield (
                                    ToolCallArgsEvent(
                                        type=EventType.TOOL_CALL_ARGS,
                                        tool_call_id=tool_call_id,
                                        delta=args,
                                    ),
                                    None,
                                    usage_metrics,
                                )
                    elif message.content:
                        # Logic for Text Content
                        if message.id != current_message_id:
                            if current_message_id:
                                yield (
                                    TextMessageEndEvent(
                                        type=EventType.TEXT_MESSAGE_END,
                                        message_id=current_message_id,
                                    ),
                                    None,
                                    usage_metrics,
                                )
                            current_message_id = message.id
                            yield (
                                TextMessageStartEvent(
                                    type=EventType.TEXT_MESSAGE_START,
                                    message_id=message.id,
                                    role="assistant",
                                ),
                                None,
                                usage_metrics,
                            )
                        yield (
                            TextMessageContentEvent(
                                type=EventType.TEXT_MESSAGE_CONTENT,
                                message_id=message.id,
                                delta=message.content,
                            ),
                            None,
                            usage_metrics,
                        )

            elif mode == "updates":
                update_event = event
                # Basic metrics accumulation from updates
                current_node = next(iter(update_event))
                node_data = update_event[current_node]
                current_usage = node_data.get("usage", {}) if node_data else {}
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

                if current_message_id:
                    yield (
                        TextMessageEndEvent(
                            type=EventType.TEXT_MESSAGE_END,
                            message_id=current_message_id,
                        ),
                        None,
                        usage_metrics,
                    )
                    current_message_id = None

        # Final yield
        yield "", None, usage_metrics

    def llm(
        self,
        preferred_model: str | None = None,
        auto_model_override: bool = True,
    ) -> ChatLiteLLM:
        """Returns the ChatLiteLLM to use for a given model.

        Args:
            preferred_model: The model to use. If none, defaults to config.llm_default_model.
            auto_model_override: If True, automatically fall back to default model
                                 if LLM Gateway is not available.

        Returns:
            ChatLiteLLM: The configured chat model.
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
        )
