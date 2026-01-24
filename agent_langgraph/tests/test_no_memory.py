import asyncio
from openai import AsyncOpenAI
import uuid

# Reusing the base URL from previous tests
BASE_URL = "http://localhost:8842"


async def test_no_thread_id():
    client = AsyncOpenAI(base_url=BASE_URL, api_key="empty")

    print("Testing Agent WITHOUT thread_id...")

    # 1. First turn: Tell the name
    # distinct name to avoid confusion with other tests
    print("\nSending Message 1 (No thread_id)...")
    response_content1 = ""
    try:
        stream = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi, my name is StatelessUser."}],
            stream=True,
            model="datarobot/azure/gpt-5-mini-2025-08-07",
            # NO extra_body provided
        )
        async for part in stream:
            if token := part.choices[0].delta.content or "":
                response_content1 += token
        print(f"Response 1: {response_content1}")
    except Exception as e:
        print(f"❌ Failed Message 1: {e}")
        return

    if not response_content1:
        print("❌ Failed: Empty response for Message 1")
        return

    # 2. Second turn: Ask for name
    # Since we didn't provide a thread_id, we expect this might fail to recall,
    # BUT the request itself should succeed (no crash).
    print("\nSending Message 2 (No thread_id)...")
    response_content2 = ""
    try:
        stream = await client.chat.completions.create(
            messages=[{"role": "user", "content": "What is my name?"}],
            stream=True,
            model="datarobot/azure/gpt-5-mini-2025-08-07",
            # NO extra_body provided
        )
        async for part in stream:
            if token := part.choices[0].delta.content or "":
                response_content2 += token
        print(f"Response 2: {response_content2}")
    except Exception as e:
        print(f"❌ Failed Message 2: {e}")
        return

    if not response_content2:
        print("❌ Failed: Empty response for Message 2")
        return

    print(
        "\n✅ NO-THREAD-ID TEST PASSED: Agent responded successfully to both requests (memory loss is expected)."
    )


if __name__ == "__main__":
    asyncio.run(test_no_thread_id())
