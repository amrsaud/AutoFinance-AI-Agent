import urllib.request
import urllib.error
import json
import uuid
import time

BASE_URL = "http://localhost:8842/chat/completions"


def post_json(url, data):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise


def test_memory_persistence():
    thread_id = str(uuid.uuid4())
    print(f"Testing with thread_id: {thread_id}")

    # 1. First turn: Tell the agent my name
    payload1 = {
        "messages": [{"role": "user", "content": "Hi, my name is Patrick."}],
        "model": "datarobot-deployed-llm",
        "extra_body": {"datarobot_association_id": thread_id},
    }

    print("\nSending Message 1...")
    try:
        result1 = post_json(BASE_URL, payload1)
        print("Response 1:", json.dumps(result1, indent=2))
    except Exception as e:
        print(f"Failed Message 1: {e}")
        return

    # 2. Second turn: Ask for the name
    payload2 = {
        "messages": [{"role": "user", "content": "What is my name?"}],
        "model": "datarobot-deployed-llm",
        "extra_body": {"datarobot_association_id": thread_id},
    }

    print("\nSending Message 2...")
    try:
        result2 = post_json(BASE_URL, payload2)
        print("Response 2:", json.dumps(result2, indent=2))

        result_str = json.dumps(result2)
        if "Patrick" in result_str:
            print("\n✅ SUBTEST PASSED: Agent remembered execution.")
        else:
            print(
                f"\n❌ SUBTEST FAILED: Agent returned '{result_str}' instead of 'Patrick'"
            )

    except Exception as e:
        print(f"Failed Message 2: {e}")


if __name__ == "__main__":
    try:
        # Simple health check on root
        try:
            # Just checking if port is open/server responds
            with urllib.request.urlopen("http://localhost:8842") as r:
                pass
        except urllib.error.HTTPError as e:
            pass
        except Exception:
            print("Server might not be running or accepting connections.")

        test_memory_persistence()
    except Exception as e:
        print(f"Test failed: {e}")
