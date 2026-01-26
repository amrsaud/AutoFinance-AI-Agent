import unittest
import uuid

from tools.supabase_client import get_supabase_client, submit_application


class TestSupabaseIntegration(unittest.TestCase):
    def setUp(self):
        # Ensure we are using actual environment variables
        # We assume .env is loaded or env vars are set
        pass

    def test_live_submission(self):
        """Test sending a real application to Supabase."""
        print("\n--- Starting Live Supabase Integration Test ---")

        # Test Data with unique session to avoid conflicts
        session_id = f"test_live_{uuid.uuid4().hex[:8]}"
        user_profile = {
            "contact_name": "Integration Test User",
            "contact_phone": "+0000000000",
            "contact_email": "integration@test.com",
            "monthly_income": 99999,
            "employment_type": "IntegrationTest",
        }
        vehicle_details = {
            "make": "TestMake",
            "model": "TestModel",
            "year": 2025,
            "price": 500000,
        }
        selected_quote = {
            "plan_name": "Test Plan",
            "monthly_installment": 5000,
            "downpayment": 100000,
            "tenure_months": 12,
            "interest_rate": 0.0,
        }

        try:
            # 1. Insert
            app_id = submit_application.invoke(
                {
                    "session_id": session_id,
                    "user_profile": user_profile,
                    "vehicle_details": vehicle_details,
                    "selected_quote": selected_quote,
                }
            )
            print(f"✅ Successfully inserted application. ID: {app_id}")
            self.assertIsNotNone(app_id)
            self.assertTrue(len(str(app_id)) > 0)

            # 2. Verify Retrieval (Optional but good)
            client = get_supabase_client()
            response = (
                client.table("applications").select("*").eq("id", app_id).execute()
            )

            self.assertTrue(len(response.data) > 0)
            record = response.data[0]
            self.assertEqual(record["session_id"], session_id)
            self.assertEqual(record["vehicle_make"], "TestMake")
            print(f"✅ Successfully verified record existence for ID: {app_id}")

            # 3. Cleanup (Delete the test record)
            # client.table("applications").delete().eq("id", app_id).execute()
            # print(f"✅ Cleaned up test record.")
            # Commenting out cleanup to let user see it if they want, or we can clean up.
            # User said "add a test", usually implies persistence check. I'll leave it or clean it?
            # Better to clean up to avoid junk, but maybe user wants to see it in dashboard.
            # I will leave cleanup commented out for now so they can verify in UI.

        except Exception as e:
            self.fail(f"Live Supabase test failed: {e}")


if __name__ == "__main__":
    unittest.main()
