"""Test memory persistence on deployed DataRobot agent."""

import asyncio
import os
import uuid

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# DataRobot deployment config from environment
DEPLOYMENT_ID = os.environ.get("AGENT_DEPLOYMENT_ID")
BASE_URL = (
    os.environ.get("DATAROBOT_ENDPOINT", "https://app.datarobot.com/api/v2")
    + "/deployments"
)
API_TOKEN = os.environ.get("DATAROBOT_API_TOKEN")


async def test_deployed_memory():
    """Test that the deployed agent remembers information across turns."""
    thread_id = str(uuid.uuid4())
    print(f"Testing deployed agent with thread_id: {thread_id}\n")

    client = AsyncOpenAI(
        base_url=f"{BASE_URL}/{DEPLOYMENT_ID}",
        api_key=API_TOKEN,
    )

    # Message 1: Tell the agent a name
    print("Sending Message 1...")
    response1 = await client.chat.completions.create(
        model="datarobot/azure/gpt-5-mini-2025-08-07",
        messages=[{"role": "user", "content": "My name is DeployedUser. Remember it."}],
        extra_body={"datarobot_association_id": thread_id},
    )
    content1 = response1.choices[0].message.content
    print(f"Response 1: {content1}\n")

    # Message 2: Ask if the agent remembers
    print("Sending Message 2...")
    response2 = await client.chat.completions.create(
        model="datarobot/azure/gpt-5-mini-2025-08-07",
        messages=[{"role": "user", "content": "What is my name?"}],
        extra_body={"datarobot_association_id": thread_id},
    )
    content2 = response2.choices[0].message.content
    print(f"Response 2: {content2}\n")

    # Check if memory worked
    if "DeployedUser" in content2:
        print("✅ DEPLOYMENT TEST PASSED: Agent remembered name.")
    else:
        print(
            f"❌ DEPLOYMENT TEST FAILED: Agent did NOT remember name. Got: '{content2}'"
        )


if __name__ == "__main__":
    asyncio.run(test_deployed_memory())
