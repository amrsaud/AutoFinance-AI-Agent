import asyncio
import uuid
import sys
from openai import AsyncOpenAI

BASE_URL = "http://localhost:8842/v1"  # Adjust if needed, typically /v1 or just base


async def test_openai_client_memory():
    # Use the same base_url logic as lit.py (config.agent_endpoint)
    # lit.py uses: client = AsyncOpenAI(base_url=config.agent_endpoint, api_key="empty")
    # config.agent_endpoint defaults to "http://localhost:8842"

    client = AsyncOpenAI(base_url="http://localhost:8842", api_key="empty")

    thread_id = str(uuid.uuid4())
    print(f"Testing AsyncOpenAI Client with thread_id: {thread_id}")

    # 1. First turn
    print("\nSending Message 1...")
    response_content = ""
    stream = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Hi, my name is OpenAIUser."}],
        stream=True,
        model="datarobot/azure/gpt-5-mini-2025-08-07",
        extra_body={"datarobot_association_id": thread_id},
    )
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            response_content += token
    print(f"Response 1: {response_content}")

    # 2. Second turn
    print("\nSending Message 2...")
    response_content = ""
    stream = await client.chat.completions.create(
        messages=[{"role": "user", "content": "What is my name?"}],
        stream=True,
        model="datarobot/azure/gpt-5-mini-2025-08-07",
        extra_body={"datarobot_association_id": thread_id},
    )
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            response_content += token
    print(f"Response 2: {response_content}")

    if "OpenAIUser" in response_content:
        print("\n✅ CLIENT TEST PASSED: Agent remembered name.")
    else:
        print(
            f"\n❌ CLIENT TEST FAILED: Agent did NOT remember name. Got: '{response_content}'"
        )


if __name__ == "__main__":
    asyncio.run(test_openai_client_memory())
