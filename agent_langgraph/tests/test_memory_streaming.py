import urllib.request
import urllib.error
import json
import uuid
import sys

BASE_URL = "http://localhost:8842/chat/completions"


def test_streaming_memory():
    thread_id = str(uuid.uuid4())
    print(f"Testing STREAMING with thread_id: {thread_id}")

    # 1. First turn: Tell the agent my name (Streamed)
    payload1 = {
        "messages": [{"role": "user", "content": "Hi, my name is StreamingUser."}],
        "model": "datarobot-deployed-llm",
        "stream": True,
        "extra_body": {"datarobot_association_id": thread_id},
    }

    print("\nSending Message 1 (Streaming)...")
    full_response1 = ""
    try:
        req = urllib.request.Request(
            BASE_URL,
            data=json.dumps(payload1).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as response:
            for line in response:
                line = line.decode("utf-8").strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line[6:])
                    if content := data["choices"][0]["delta"].get("content"):
                        full_response1 += content
        print(f"Response 1: {full_response1}")
    except Exception as e:
        print(f"Failed Message 1: {e}")
        return

    # 2. Second turn: Ask for the name (Streamed)
    payload2 = {
        "messages": [{"role": "user", "content": "What is my name?"}],
        "model": "datarobot-deployed-llm",
        "stream": True,
        "extra_body": {"datarobot_association_id": thread_id},
    }

    print("\nSending Message 2 (Streaming)...")
    full_response2 = ""
    try:
        req = urllib.request.Request(
            BASE_URL,
            data=json.dumps(payload2).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as response:
            for line in response:
                line = line.decode("utf-8").strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line[6:])
                    if content := data["choices"][0]["delta"].get("content"):
                        full_response2 += content
        print(f"Response 2: {full_response2}")

        if "StreamingUser" in full_response2:
            print("\n✅ STREAMING TEST PASSED: Agent remembered name.")
        else:
            print(
                f"\n❌ STREAMING TEST FAILED: Agent did NOT remember name. Got: '{full_response2}'"
            )

    except Exception as e:
        print(f"Failed Message 2: {e}")


if __name__ == "__main__":
    test_streaming_memory()
