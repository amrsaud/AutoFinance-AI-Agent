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
Conversational agent with memory persistence using LangGraph checkpointing.
Remembers user details (name, age) across conversations.
"""

import uuid
from typing import Any

from config import Config
from datarobot_genai.core.agents import make_system_prompt
from datarobot_genai.langgraph.agent import LangGraphAgent
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_litellm.chat_models import ChatLiteLLM
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from psycopg import AsyncConnection

config = Config()


class MyAgent(LangGraphAgent):
    """Conversational agent with memory that remembers user details.

    Uses LangGraph checkpointing with PostgreSQL for persistent memory
    across conversations. Tracks user information like name and age.
    """

    def __init__(
        self,
        thread_id: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the agent with optional thread_id for conversation tracking.

        Args:
            thread_id: Unique identifier for the conversation thread.
                       If not provided, a new UUID will be generated.
            **kwargs: Additional arguments passed to LangGraphAgent.
        """
        super().__init__(**kwargs)
        self.thread_id = thread_id or str(uuid.uuid4())
        # We don't initialize checkpointer here, we limit its scope to the request

    @property
    def checkpointer(self) -> None:
        """Deprecated: Checkpointer is managed in _invoke."""
        return None

    @property
    def langgraph_config(self) -> dict[str, Any]:
        """Configuration for LangGraph including thread_id for checkpointing."""
        return {"configurable": {"thread_id": self.thread_id}}

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
        """Build the conversational agent workflow.

        Returns a react agent which handles multi-turn conversations correctly.
        """
        # Create a react agent - this returns a CompiledGraph
        # We need to re-compile it with the checkpointer at runtime
        # So we reconstruct the graph here

        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self._call_model)
        workflow.add_edge(START, "agent")
        workflow.add_edge("agent", END)

        return workflow

    async def _call_model(self, state: MessagesState):
        """Invoke the LLM with the current state."""
        system_prompt = make_system_prompt(
            "You are a friendly and helpful conversational assistant.\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "1. When a user tells you their name or age, remember it for future conversations.\n"
            "2. If you know the user's name, greet them by name in responses.\n"
            "3. Keep track of personal details shared by the user.\n"
            "4. If asked about previously shared information, recall it accurately.\n"
            "5. Be natural and conversational in your responses.\n\n"
            "Example interactions:\n"
            "- User: 'My name is John' -> Remember and acknowledge\n"
            "- User: 'I'm 25 years old' -> Remember the age\n"
            "- User: 'What's my name?' -> Recall 'John' if previously shared\n"
        )

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = await self.llm(
            preferred_model="datarobot/azure/gpt-5-mini-2025-08-07"
        ).ainvoke(messages)
        return {"messages": [response]}

    async def _invoke(self, completion_create_params):
        """Override to compile workflow with checkpointer for memory persistence."""
        from datarobot_genai.langgraph.agent import is_streaming

        input_command = self.convert_input_message(completion_create_params)

        # Allow thread_id to be passed via 'extra_body' using 'thread_id' or 'datarobot_association_id'
        # We generally expect these to be in extra_body for OpenAI-compatible requests
        extra_body = completion_create_params.get("extra_body") or {}
        request_thread_id = (
            extra_body.get("thread_id")
            or extra_body.get("datarobot_association_id")
            or completion_create_params.get("datarobot_association_id")
            or self.thread_id
        )
        run_config = {"configurable": {"thread_id": request_thread_id}}

        if config.postgres_uri:
            # Check if streaming is requested
            if is_streaming(completion_create_params):
                return self._stream_with_db(
                    input_command,
                    completion_create_params,
                    run_config,
                )
            else:
                # Non-streaming: execute within the context
                async with await AsyncConnection.connect(
                    config.postgres_uri, autocommit=True
                ) as conn:
                    # Disable prepared statements for Supabase transaction mode
                    conn.prepare_threshold = 0
                    try:
                        async with conn.cursor() as cur:
                            await cur.execute("DEALLOCATE ALL")
                    except Exception:
                        pass

                    checkpointer = AsyncPostgresSaver(conn)
                    try:
                        await checkpointer.setup()
                    except Exception:
                        pass

                    agent_graph = self.workflow.compile(checkpointer=checkpointer)

                    return await self._execute_graph(
                        input_command,
                        completion_create_params,
                        agent_graph,
                        is_streaming,
                        run_config,
                    )
        else:
            # Fallback without persistence
            return await self._execute_graph(
                input_command,
                completion_create_params,
                self.workflow.compile(),
                is_streaming,
                run_config,
            )

    async def _stream_with_db(
        self, input_command, completion_create_params, run_config
    ):
        """Helper to stream response while keeping DB connection open."""
        from datarobot_genai.langgraph.agent import is_streaming

        try:
            async with await AsyncConnection.connect(
                config.postgres_uri, autocommit=True
            ) as conn:
                conn.prepare_threshold = 0
                try:
                    async with conn.cursor() as cur:
                        await cur.execute("DEALLOCATE ALL")
                except Exception:
                    pass

                checkpointer = AsyncPostgresSaver(conn)
                try:
                    await checkpointer.setup()
                except Exception:
                    pass

                agent_graph = self.workflow.compile(checkpointer=checkpointer)

                # Get the generator from _execute_graph
                # Note: _execute_graph returns the generator immediately for streaming
                generator = await self._execute_graph(
                    input_command,
                    completion_create_params,
                    agent_graph,
                    is_streaming,
                    run_config,
                )

                # We must iterate over the generator HERE, inside the async with block
                async for item in generator:
                    yield item

        except Exception:
            raise

    async def _execute_graph(
        self,
        input_command,
        completion_create_params,
        compiled_graph,
        is_streaming,
        run_config,
    ):
        """Execute the compiled graph."""

        # Unwrap Command object to ensure we trigger a new run from START
        graph_input = input_command
        if hasattr(input_command, "update"):
            graph_input = input_command.update

        graph_stream = compiled_graph.astream(
            input=graph_input,
            config=run_config,
            debug=True,
            stream_mode=["messages", "updates"],
            subgraphs=True,
        )

        usage_metrics = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        }

        if is_streaming(completion_create_params):
            return self._stream_generator(graph_stream, usage_metrics)
        else:
            events = []
            async for event in graph_stream:
                # print(f"DEBUG: Event: {event}")
                if isinstance(event, dict):  # values or updates
                    events.append(event)
                elif isinstance(event, tuple):  # (namespace, mode, payload)
                    if len(event) == 3:
                        namespace, mode, payload = event
                    elif len(event) == 2:
                        mode, payload = event
                        namespace = None
                    else:
                        continue

                    if mode == "updates":
                        events.append(payload)

            # print(f"DEBUG: Collected {len(events)} events.")

            for update in events:
                current_node = next(iter(update))
                node_data = update[current_node]
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

            pipeline_interactions = self.create_pipeline_interactions_from_events(
                events
            )

            # Extract response - react agent final response is usually the last message
            if events:
                # Find the last message
                # For react agent, we look for the last 'agent' or 'chatbot' update
                # Or just use aget_state() to get the final state
                pass

            # Robust way to get the final message from state
            state = await compiled_graph.aget_state(run_config)
            messages = state.values.get("messages", [])
            response_text = str(messages[-1].content) if messages else ""

            return response_text, pipeline_interactions, usage_metrics

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
