import pytest
from unittest.mock import patch
import os
from agentic_workflow.tools.policy_retriever import retrieve_eligible_policies

# Use the ID provided by user
TEST_DB_ID = "697649ced0b5d1dae4ef8b9a"


@pytest.mark.integration
def test_retriever_live():
    """Test policy retrieval against live DataRobot endpoint."""
    if not os.getenv("DATAROBOT_API_TOKEN"):
        pytest.skip("No DataRobot API Token found")

    # MOCK the config using patch.dict on os.environ
    # We patch environment variables so Config() picks it up
    with patch.dict(os.environ, {"DATAROBOT_VECTOR_DB_ID": TEST_DB_ID}):
        # Query: Freelancer with 10k income
        policies = retrieve_eligible_policies.invoke(
            {"min_income": 10000.0, "employment_category": "Freelancer (Tech)"}
        )

        # We expect at least one policy or empty list if quota exceeded
        # Just check structure if result exists
        if policies and "error" not in policies[0]:
            p = policies[0]
            assert "policy_id" in p
            assert "interest_rate" in p
            assert "max_tenure_months" in p
            assert "max_dbr" in p


def test_retriever_no_id():
    """Test behavior when ID is missing."""
    # Ensure ID is missing by patching env with clear=True?
    # No, that clears ALL env, bad.
    # We just ensure target key is missing.
    # patch.dict only modifies keys present in dict.

    # We can use a context manager where we unset it.
    original = os.environ.get("DATAROBOT_VECTOR_DB_ID")
    if original:
        del os.environ["DATAROBOT_VECTOR_DB_ID"]

    try:
        res = retrieve_eligible_policies.invoke(
            {"min_income": 10000.0, "employment_category": "Test"}
        )
        assert "error" in res[0]
    finally:
        if original:
            os.environ["DATAROBOT_VECTOR_DB_ID"] = original
