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
"""Tests for complete search flows."""

import pytest
import requests
from openai import OpenAI

BASE_URL = "http://localhost:8842"


class TestSearchFlowWithConfirmation:
    """Tests for search flow where user confirms."""

    def test_search_request_asks_for_confirmation(self):
        """Test that search request asks for confirmation."""
        thread_id = "test-confirm-ask"
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [
                    {"role": "user", "content": "I want a 2024 Hyundai Tucson"}
                ],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]["content"]
        assert any(
            word in msg.lower()
            for word in ["confirm", "proceed", "search", "yes", "hyundai"]
        )

    def test_search_confirmation_returns_results(self):
        """Test that confirming search returns results."""
        thread_id = "test-confirm-results"

        # Step 1: Request search
        requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "Find a Toyota Camry 2023"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )

        # Step 2: Confirm
        r2 = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "yes, search for it"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r2.status_code == 200
        msg = r2.json()["choices"][0]["message"]["content"]
        assert any(
            word in msg.lower()
            for word in ["found", "vehicle", "result", "toyota", "no"]
        )


class TestSearchFlowWithCancellation:
    """Tests for search flow where user cancels."""

    def test_search_cancellation_acknowledged(self):
        """Test that cancelling search is acknowledged."""
        thread_id = "test-cancel-ack"

        # Step 1: Request search
        requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "Find a Kia Sportage"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )

        # Step 2: Cancel
        r2 = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "no, never mind"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r2.status_code == 200
        msg = r2.json()["choices"][0]["message"]["content"]
        assert msg  # Should have some response


class TestSearchParameterVariations:
    """Tests for different search parameter combinations."""

    def test_search_with_make_only(self):
        """Test search with only car make specified."""
        thread_id = "test-make-only-param"
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "I want a Mercedes"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]["content"]
        assert "mercedes" in msg.lower()

    def test_search_with_make_and_model(self):
        """Test search with make and model specified."""
        thread_id = "test-make-model-param"
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "Find a BMW X5"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]["content"]
        assert "bmw" in msg.lower()

    def test_search_with_year_constraint(self):
        """Test search with year specified."""
        thread_id = "test-with-year-param"
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "user", "content": "I want a 2023 Kia Sportage"}],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]["content"]
        assert "2023" in msg or "kia" in msg.lower()

    def test_search_with_price_constraint(self):
        """Test search with price limit specified."""
        thread_id = "test-with-price-param"
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [
                    {"role": "user", "content": "Find a car under 500000 EGP"}
                ],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]["content"]
        assert "500" in msg or "price" in msg.lower() or "egp" in msg.lower() or msg

    def test_search_with_all_parameters(self):
        """Test search with make, model, year, and price."""
        thread_id = "test-all-params-flow"
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": "agent",
                "messages": [
                    {
                        "role": "user",
                        "content": "I want a 2024 Hyundai Tucson under 800000 EGP",
                    }
                ],
                "extra_body": {"thread_id": thread_id},
                "stream": False,
            },
        )
        assert r.status_code == 200
        msg = r.json()["choices"][0]["message"]["content"]
        assert "hyundai" in msg.lower()


class TestSearchViaOpenAIClient:
    """Tests using OpenAI client."""

    def test_search_request_via_openai_client(self):
        """Test search request using OpenAI client."""
        client = OpenAI(base_url=BASE_URL, api_key="dummy")
        r = client.chat.completions.create(
            model="agent",
            messages=[{"role": "user", "content": "I want a 2024 Hyundai Elantra"}],
            extra_body={"thread_id": "test-openai-search-req"},
        )
        assert r.choices[0].message.content
        assert "hyundai" in r.choices[0].message.content.lower()

    def test_complete_search_flow_via_openai_client(self):
        """Test complete search flow using OpenAI client."""
        client = OpenAI(base_url=BASE_URL, api_key="dummy")
        thread_id = "test-openai-complete-flow"

        # Step 1: Search request
        r1 = client.chat.completions.create(
            model="agent",
            messages=[{"role": "user", "content": "Find a Mazda CX-5"}],
            extra_body={"thread_id": thread_id},
        )
        assert r1.choices[0].message.content

        # Step 2: Confirm
        r2 = client.chat.completions.create(
            model="agent",
            messages=[{"role": "user", "content": "yes please"}],
            extra_body={"thread_id": thread_id},
        )
        assert r2.choices[0].message.content
