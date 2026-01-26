import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage
from models import UserProfile, Vehicle, LoanQuote, EmploymentType
from nodes.submission import submission_node


class TestSubmissionNode(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Mock State
        self.user_profile = UserProfile(
            monthly_income=50000,
            employment_type=EmploymentType.SALARIED_PRIVATE,
            contact_name="Test User",
            contact_phone="+123",
            contact_email="test@test.com",
        )
        self.vehicle = Vehicle(
            make="Toyota",
            model="Corolla",
            price=1000000,
            source="hatla2ee",
            source_url="http://test",
        )
        self.quotes = [
            LoanQuote(
                policy_id="pol1",
                plan_name="Plan A",
                monthly_installment=10000,
                tenure_months=60,
                interest_rate=15.0,
                is_affordable=True,
                dbr_percentage=20.0,
            ),
            LoanQuote(
                policy_id="pol2",
                plan_name="Plan B",
                monthly_installment=15000,
                tenure_months=36,
                interest_rate=14.0,
                is_affordable=True,
                dbr_percentage=30.0,
            ),
        ]

    @patch("nodes.submission.submit_application")
    async def test_submission_select_option_success(self, mock_submit):
        # Setup LLM Mock
        mock_llm = MagicMock()
        mock_structured_llm = AsyncMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        # Determine Return Type of structured LLM
        # It should return a Pydantic object
        mock_decision = MagicMock()
        mock_decision.decision = "select_option"
        mock_decision.option_number = 1
        mock_structured_llm.ainvoke.return_value = mock_decision

        mock_submit.invoke.return_value = "APP-123"

        state = {
            "messages": [HumanMessage(content="I choose Option 1")],
            "user_profile": self.user_profile,
            "selected_vehicle": self.vehicle,
            "generated_quotes": self.quotes,
            "awaiting_submission": True,
        }

        # Execute
        result = await submission_node(state, mock_llm)

        # Verify LLM Called
        mock_llm.with_structured_output.assert_called()
        mock_structured_llm.ainvoke.assert_called()

        # Verify Submit Tool Called
        mock_submit.invoke.assert_called_once()
        # With invoke, the first arg is the dictionary
        # args[0] is the dict. kwargs might be empty depending on how invoke is called.
        args, _ = mock_submit.invoke.call_args
        submitted_data = args[0]
        self.assertEqual(submitted_data["selected_quote"]["plan_name"], "Plan A")

        # Verify Response
        messages = result.get("messages")
        self.assertTrue("APP-123" in messages[0].content)
        self.assertFalse(result.get("awaiting_submission"))

    @patch("nodes.submission.submit_application")
    async def test_submission_start_over(self, mock_submit):
        # Setup LLM Mock
        mock_llm = MagicMock()
        mock_structured_llm = AsyncMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        mock_decision = MagicMock()
        mock_decision.decision = "start_over"
        mock_decision.option_number = None
        mock_structured_llm.ainvoke.return_value = mock_decision

        state = {
            "messages": [HumanMessage(content="Start over please")],
            "user_profile": self.user_profile,
            "selected_vehicle": self.vehicle,
            "generated_quotes": self.quotes,
        }

        # Execute
        result = await submission_node(state, mock_llm)

        # Verify Submit NOT Called
        mock_submit.invoke.assert_not_called()

        # Verify Reset
        self.assertIsNone(result.get("selected_vehicle"))
        self.assertFalse(result.get("awaiting_submission"))
        self.assertTrue("start over" in result["messages"][0].content.lower())

    @patch("nodes.submission.submit_application")
    async def test_submission_unknown_intent(self, mock_submit):
        # Setup LLM Mock
        mock_llm = MagicMock()
        mock_structured_llm = AsyncMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        mock_decision = MagicMock()
        mock_decision.decision = "unknown"
        mock_structured_llm.ainvoke.return_value = mock_decision

        state = {
            "messages": [HumanMessage(content="What?")],
            "user_profile": self.user_profile,
            "selected_vehicle": self.vehicle,
            "generated_quotes": self.quotes,
        }

        # Execute
        result = await submission_node(state, mock_llm)

        # Verify
        mock_submit.invoke.assert_not_called()
        self.assertTrue("not sure" in result["messages"][0].content.lower())
