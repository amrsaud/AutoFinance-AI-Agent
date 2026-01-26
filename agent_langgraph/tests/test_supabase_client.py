import unittest
from unittest.mock import patch, MagicMock
from tools.supabase_client import submit_application


import tools.supabase_client


class TestSupabaseClient(unittest.TestCase):
    def setUp(self):
        # Reset global client to ensure create_client is called
        tools.supabase_client._supabase_client = None

    @patch("tools.supabase_client.create_client")
    @patch("tools.supabase_client.Config")
    def test_submit_application_success(self, MockConfig, mock_create_client):
        # Setup Mock Config
        mock_config_instance = MockConfig.return_value
        mock_config_instance.supabase_url = "https://test.supabase.co"
        mock_config_instance.supabase_key = "test_key"

        # Setup Mock Client
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        # Setup Insert Response
        mock_response = MagicMock()
        mock_response.data = [{"id": "test-uuid-123"}]
        # Ensure the chain returns this response
        # client.table("applications").insert(payload).execute()
        # Mock: client.table.return_value.insert.return_value.execute.return_value
        table_mock = mock_client.table.return_value
        insert_mock = table_mock.insert.return_value
        insert_mock.execute.return_value = mock_response

        # Test Data
        session_id = "sess_001"
        user_profile = {
            "contact_name": "Test User",
            "contact_phone": "+1234567890",
            "contact_email": "test@example.com",
            "monthly_income": 50000,
            "employment_type": "Salaried",
        }
        vehicle_details = {
            "make": "Hyundai",
            "model": "Tucson",
            "year": 2024,
            "price": 1500000,
        }
        selected_quote = {
            "plan_name": "Standard Plan",
            "monthly_installment": 12000,
            "downpayment": 300000,
            "tenure_months": 60,
            "interest_rate": 15.0,
        }

        # Execute
        result_id = submit_application(
            session_id, user_profile, vehicle_details, selected_quote
        )

        # Verify
        self.assertEqual(result_id, "test-uuid-123")

        # Verify call arguments (flattening check)
        expected_payload = {
            "session_id": session_id,
            "user_name": "Test User",
            "contact_phone": "+1234567890",
            "contact_email": "test@example.com",
            "monthly_income": 50000,
            "employment_type": "Salaried",
            "vehicle_make": "Hyundai",
            "vehicle_model": "Tucson",
            "vehicle_year": 2024,
            "vehicle_price": 1500000,
            "quote_plan_name": "Standard Plan",
            "quote_monthly_installment": 12000,
            "quote_downpayment": 300000,
            "quote_tenure": 60,
            "quote_interest_rate": 15.0,
            "status": "pending_review",
        }
        mock_client.table.assert_called_with("applications")
        mock_client.table.return_value.insert.assert_called_with(expected_payload)

    @patch("tools.supabase_client.create_client")
    @patch("tools.supabase_client.Config")
    def test_submit_application_failure(self, MockConfig, mock_create_client):
        # Setup Mock
        mock_config_instance = MockConfig.return_value
        mock_config_instance.supabase_url = "https://test.supabase.co"
        mock_config_instance.supabase_key = "test_key"

        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        # Simulate Empty Response
        mock_response = MagicMock()
        mock_response.data = []
        mock_client.table.return_value.insert.return_value.execute.return_value = (
            mock_response
        )

        # Execute & Expect Error
        with self.assertRaises(Exception) as context:
            submit_application("sess_001", {}, {}, {})

        self.assertTrue("No data returned" in str(context.exception))
