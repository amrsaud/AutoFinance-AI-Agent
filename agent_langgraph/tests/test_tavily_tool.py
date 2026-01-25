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
"""Tests for Tavily search tool."""

import os

import pytest

# Skip all tests in this module if TAVILY_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("TAVILY_API_KEY"),
    reason="TAVILY_API_KEY environment variable not set",
)


class TestTavilySearchTool:
    """Direct tests for Tavily search tool functionality."""

    def test_tavily_search_returns_results(self):
        """Test that Tavily search returns non-empty results."""
        from agentic_workflow.tools.tavily_search import search_egyptian_cars

        result = search_egyptian_cars.invoke("Hyundai Tucson 2024 Egypt")
        assert result
        assert len(result) > 0

    def test_tavily_search_targets_egyptian_sites(self):
        """Test that search targets Egyptian car marketplaces."""
        from agentic_workflow.tools.tavily_search import search_egyptian_cars

        result = search_egyptian_cars.invoke("Toyota Corolla Egypt")
        result_lower = result.lower()
        # Should include Egyptian marketplace references
        assert (
            "hatla2ee" in result_lower
            or "dubizzle" in result_lower
            or "egypt" in result_lower
        )

    def test_tavily_search_with_make_only(self):
        """Test search with only car make."""
        from agentic_workflow.tools.tavily_search import search_egyptian_cars

        result = search_egyptian_cars.invoke("Mercedes Egypt")
        assert result
        assert len(result) > 0

    def test_tavily_search_with_make_model_year(self):
        """Test search with make, model, and year."""
        from agentic_workflow.tools.tavily_search import search_egyptian_cars

        result = search_egyptian_cars.invoke("Honda Civic 2023 Egypt")
        assert result
        assert len(result) > 0

    def test_tavily_search_with_bmw(self):
        """Test search for BMW vehicles."""
        from agentic_workflow.tools.tavily_search import search_egyptian_cars

        result = search_egyptian_cars.invoke("BMW X5 Cairo Egypt")
        assert result
        assert len(result) > 0

    def test_tavily_search_handles_empty_query(self):
        """Test that search handles empty query gracefully."""
        from agentic_workflow.tools.tavily_search import search_egyptian_cars

        # Even with empty query, should not raise exception
        result = search_egyptian_cars.invoke("")
        assert result is not None  # Should return something (may be error or empty)
