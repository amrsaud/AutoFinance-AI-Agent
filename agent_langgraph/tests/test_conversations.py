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
"""Tests for conversations and edge cases."""

import pytest
import requests
from openai import OpenAI

BASE_URL = "http://localhost:8842"


class TestGeneralConversation:
    """Tests for general conversation without search intent."""

    def test_greeting_response(self):
        """Test agent responds to greeting."""
        thread_id = "test-greeting-conv"
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "Hello!"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"]

    def test_capability_question(self):
        """Test agent explains its capabilities."""
        thread_id = "test-capabilities-conv"
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "What can you help me with?"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]["content"]
        assert any(
            word in msg.lower() for word in ["car", "vehicle", "search", "find", "help"]
        )

    def test_follow_up_question(self):
        """Test agent handles follow-up questions."""
        thread_id = "test-followup-conv"
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [
                    {"role": "user", "content": "What websites do you search?"}
                ],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]["content"]
        assert any(
            word in msg.lower()
            for word in ["hatla2ee", "dubizzle", "website", "marketplace", "egypt"]
        )


class TestMultiTurnConversation:
    """Tests for multi-turn conversations."""

    def test_conversation_context_retained(self):
        """Test that conversation context is retained across turns."""
        thread_id = "test-multi-turn-conv"

        # Turn 1
        r1 = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [
                    {"role": "user", "content": "Hi, what cars can you find?"}
                ],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r1.status_code == 200

        # Turn 2
        r2 = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [
                    {"role": "user", "content": "Great, find me a Mazda CX-5"}
                ],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r2.status_code == 200
        msg = r2.json()["choices"][0]["message"]["content"]
        assert "mazda" in msg.lower()

    def test_three_turn_conversation(self):
        """Test three-turn conversation flow."""
        thread_id = "test-three-turns-conv"

        # Turn 1: Greeting
        r1 = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "Hello"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r1.status_code == 200

        # Turn 2: Ask capability
        r2 = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "Can you find cars for me?"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r2.status_code == 200

        # Turn 3: Search
        r3 = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "Find a Honda Accord"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r3.status_code == 200


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_search_results_handled(self):
        """Test handling when no vehicles found."""
        thread_id = "test-empty-results-edge"

        # Search for something unlikely
        r1 = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [
                    {"role": "user", "content": "Find a Rolls Royce Phantom 2025"}
                ],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        r2 = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "yes"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r2.status_code == 200

    def test_thread_id_isolation(self):
        """Test that different thread_ids have isolated state."""
        # Thread A: Search Toyota
        requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "Find a Toyota RAV4"}],
                "extra_body": {"thread_id": "thread-isolation-a-edge"},
                "stream": False,
            },
        )

        # Thread B: Search Honda (different thread)
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "Find a Honda CR-V"}],
                "extra_body": {"thread_id": "thread-isolation-b-edge"},
                "stream": False,
            },
        )
        msg = r.json()["choices"][0]["message"]["content"]
        assert "honda" in msg.lower()

    def test_special_characters_in_query(self):
        """Test handling of special characters in search query."""
        thread_id = "test-special-chars-edge"
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "Find a BMW 3-Series (2024)"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r.status_code == 200

    def test_arabic_location_in_query(self):
        """Test handling of Arabic text in query."""
        thread_id = "test-arabic-edge"
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "Find a car in القاهرة"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r.status_code == 200
