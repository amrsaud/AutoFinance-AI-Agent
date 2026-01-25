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
"""Tests for state reset functionality."""

import pytest
import requests
from openai import OpenAI

BASE_URL = "http://localhost:8842"


class TestResetAtStart:
    """Tests for reset when conversation just started."""

    def test_reset_start_over_command(self):
        """Test 'start over' command resets state."""
        thread_id = "test-reset-start-over-cmd"
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "start over"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]["content"]
        assert any(
            word in msg.lower() for word in ["fresh", "clear", "start", "help", "find"]
        )

    def test_reset_clear_everything_command(self):
        """Test 'clear everything' command resets state."""
        thread_id = "test-reset-clear-cmd"
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "clear everything"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]["content"]
        assert any(
            word in msg.lower() for word in ["fresh", "clear", "start", "help", "find"]
        )

    def test_reset_begin_fresh_command(self):
        """Test 'begin fresh' command resets state."""
        thread_id = "test-reset-fresh-cmd"
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "let's begin fresh"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r.status_code == 200


class TestResetMidFlow:
    """Tests for reset during search flow."""

    def test_reset_after_search_params_extracted(self):
        """Test reset after search params extracted but before confirmation."""
        thread_id = "test-reset-mid-flow"

        # Step 1: Start search
        requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "Find a Toyota Corolla"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )

        # Step 2: Reset instead of confirming
        r2 = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "actually, let's start over"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r2.status_code == 200
        msg = r2.json()["choices"][0]["message"]["content"]
        assert any(
            word in msg.lower() for word in ["fresh", "clear", "start", "reset", "help"]
        )

    def test_new_search_after_reset_has_no_old_context(self):
        """Test that new search after reset doesn't have old context."""
        thread_id = "test-reset-no-context-flow"

        # Step 1: Search Toyota
        requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "Find a Toyota Corolla"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )

        # Step 2: Reset
        requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "start over"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )

        # Step 3: New search for Honda
        r3 = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "Now find me a Honda Civic"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r3.status_code == 200
        msg = r3.json()["choices"][0]["message"]["content"]
        assert "honda" in msg.lower()


class TestResetAfterSearchComplete:
    """Tests for reset after search is complete."""

    def test_reset_after_receiving_results(self):
        """Test reset after search results received."""
        thread_id = "test-reset-after-results-flow"

        # Complete a full search
        requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "Find a Nissan Sunny"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "yes"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )

        # Reset
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [
                    {"role": "user", "content": "clear everything and start fresh"}
                ],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]["content"]
        assert any(word in msg.lower() for word in ["fresh", "clear", "start"])


class TestResetViaOpenAIClient:
    """Tests for reset using OpenAI client."""

    def test_reset_via_openai_client(self):
        """Test reset using OpenAI client."""
        client = OpenAI(base_url=BASE_URL, api_key="dummy")
        thread_id = "test-openai-reset-flow"

        # Start search
        client.chat.completions.create(
            model="agent",
            messages=[{"role": "user", "content": "Find a Chevrolet Optra"}],
            extra_body={"thread_id": thread_id},
        )

        # Reset
        r = client.chat.completions.create(
            model="agent",
            messages=[{"role": "user", "content": "start over"}],
            extra_body={"thread_id": thread_id},
        )
        assert any(
            word in r.choices[0].message.content.lower()
            for word in ["fresh", "clear", "start", "help"]
        )
