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
# ------------------------------------------------------------------------------
# THIS SECTION OF CODE IS REQUIRED TO SETUP TRACING AND TELEMETRY FOR THE AGENTS.
# REMOVING THIS CODE WILL DISABLE ALL MONITORING, TRACING AND TELEMETRY.
# isort: off
from datarobot_genai.core.telemetry_agent import instrument

instrument(framework="langgraph")
# ruff: noqa: E402
from agent import MyAgent
from config import Config

# isort: on
# ------------------------------------------------------------------------------
import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, Iterator, Optional, Union

from datarobot_genai.core.chat import (
    CustomModelChatResponse,
    CustomModelStreamingResponse,
    resolve_authorization_context,
    to_custom_model_chat_response,
    to_custom_model_streaming_response,
)
from openai.types.chat import CompletionCreateParams
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming,
)

# Import checkpointer creation function
from persistence.supabase_checkpointer import create_supabase_checkpointer


def load_model(
    code_dir: str,
) -> tuple[ThreadPoolExecutor, asyncio.AbstractEventLoop, Any]:
    """
    The agent is instantiated in this function and returned.

    Also initializes the PostgreSQL checkpointer for state persistence.
    This enables multi-turn conversations with memory across requests.
    """
    thread_pool_executor = ThreadPoolExecutor(1)
    event_loop = asyncio.new_event_loop()
    thread_pool_executor.submit(asyncio.set_event_loop, event_loop).result()

    # Initialize checkpointer once at startup for reuse across requests
    checkpointer = create_supabase_checkpointer()

    return (thread_pool_executor, event_loop, checkpointer)


def _extract_thread_id(completion_create_params: dict) -> str:
    """
    Extract or generate a thread_id for conversation memory.

    Priority:
    1. Look for thread_id in extra_body
    2. Look for x-conversation-id in headers
    3. Generate a new UUID
    """
    # Check extra_body for explicit thread_id
    extra_body = completion_create_params.get("extra_body", {}) or {}
    if "thread_id" in extra_body:
        return str(extra_body["thread_id"])

    # Check for conversation ID in DataRobot association
    if "datarobot_association_id" in completion_create_params:
        return str(completion_create_params["datarobot_association_id"])

    # Check forwarded headers
    headers = completion_create_params.get("forwarded_headers", {}) or {}
    for header_name in ["x-conversation-id", "x-thread-id", "x-session-id"]:
        if header_name in headers:
            return str(headers[header_name])

    # Generate a new thread_id if none found
    # Note: This means each request without a thread_id is a new conversation
    return str(uuid.uuid4())


def chat(
    completion_create_params: CompletionCreateParams
    | CompletionCreateParamsNonStreaming
    | CompletionCreateParamsStreaming,
    load_model_result: tuple[ThreadPoolExecutor, asyncio.AbstractEventLoop, Any],
    **kwargs: Any,
) -> Union[CustomModelChatResponse, Iterator[CustomModelStreamingResponse]]:
    """When using the chat endpoint, this function is called.

    Agent inputs are in OpenAI message format and defined as the 'user' portion
    of the input prompt.

    Example:
        prompt = {
            "topic": "Artificial Intelligence",
        }
        client = OpenAI(...)
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{json.dumps(prompt)}"},
            ],
            extra_body = {
                "environment_var": True,
                "thread_id": "unique-conversation-id",  # For memory persistence
            },
            ...
        )
    """
    thread_pool_executor, event_loop, checkpointer = load_model_result

    # Change working directory to the directory containing this file.
    # Some agent frameworks expect this for expected pathing.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Load MCP runtime parameters and session secret if configured
    # ["EXTERNAL_MCP_URL", "MCP_DEPLOYMENT_ID", "SESSION_SECRET_KEY"]
    _ = Config()

    # Initialize the authorization context for downstream agents and tools to retrieve
    # access tokens for external services.
    completion_create_params["authorization_context"] = resolve_authorization_context(
        completion_create_params, **kwargs
    )

    # The list of the headers to forward into the Agent and MCP Server.
    incoming_headers = kwargs.get("headers", {}) or {}
    allowed_headers = {"x-datarobot-api-token", "x-datarobot-api-key"}
    forwarded_headers = {
        k: v for k, v in incoming_headers.items() if k.lower() in allowed_headers
    }
    completion_create_params["forwarded_headers"] = forwarded_headers

    # Extract thread_id for conversation memory
    thread_id = _extract_thread_id(completion_create_params)

    # Pass checkpointer and thread_id config to the agent
    completion_create_params["checkpointer"] = checkpointer
    completion_create_params["thread_id"] = thread_id

    # Instantiate the agent, all fields from the completion_create_params are passed to the agent
    # allowing environment variables to be passed during execution
    agent = MyAgent(**completion_create_params)

    # Invoke the agent
    result = thread_pool_executor.submit(
        event_loop.run_until_complete,
        agent.invoke(completion_create_params=completion_create_params),
    ).result()

    # Check if the result is a generator (streaming response)
    if isinstance(result, AsyncGenerator):
        # Streaming response
        return to_custom_model_streaming_response(
            thread_pool_executor,
            event_loop,
            result,
            model=completion_create_params.get("model"),
        )
    else:
        # Non-streaming response
        response_text, pipeline_interactions, usage_metrics = result

        return to_custom_model_chat_response(
            response_text,
            pipeline_interactions,
            usage_metrics,
            model=completion_create_params.get("model"),
        )
