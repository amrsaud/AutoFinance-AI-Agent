import asyncio
import os
import sys

# Add project root to path (agent_langgraph directory)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import using package name directly as we are running from root
try:
    from agentic_workflow.tools.tavily_search import search_vehicles
    from agentic_workflow.tools.supabase_client import (
        save_application,
        check_application_status,
    )
    from agentic_workflow.state import (
        SearchParams,
        CustomerInfo,
        Vehicle,
        FinancialQuote as Quote,
    )
except ImportError:
    # Fallback if we are in a different context structure
    from agent_langgraph.agentic_workflow.tools.tavily_search import search_vehicles
    from agent_langgraph.agentic_workflow.tools.supabase_client import (
        save_application,
        check_application_status,
    )
    from agent_langgraph.agentic_workflow.state import (
        SearchParams,
        CustomerInfo,
        Vehicle,
        FinancialQuote as Quote,
    )


async def test_search_tool():
    print("\n🔍 Testing Search Tool...")
    params = {
        "make": "Toyota",
        "model": "Corolla",
        "year_from": 2020,
        "year_to": 2024,
        "price_cap": 2000000,
    }

    try:
        # Tool signature: def search_vehicles(params: dict) -> list[dict]
        # Invocation: .invoke({"params": {...}})
        results = search_vehicles.invoke({"params": params})

        print(f"✅ Search successful! Found {len(results)} results.")
        if results:
            first = results[0]
            print(
                f"   Sample: {first.get('title', 'No Title')} - {first.get('url', 'No URL')}"
            )
        else:
            print("⚠️ No results found.")

    except Exception as e:
        print(f"❌ Search failed: {e}")


async def test_supabase_tools():
    print("\n💾 Testing Supabase Tools...")

    customer = CustomerInfo(
        full_name="Test User",
        email="test@example.com",
        phone="+201000000000",
        national_id="12345678901234",
    )

    vehicle = Vehicle(
        make="Hyundai",
        model="Tucson",
        year=2023,
        price=1500000,
        mileage=10000,
        source_url="http://example.com",
        source_name="TestDealer",
    )

    quote = Quote(
        principal=1500000,
        interest_rate=18.0,
        tenure_months=60,
        monthly_installment=35000,
        total_payment=2100000,
        total_interest=600000,
    )

    print("   Testing save_application...")
    try:
        # Tool: def save_application(customer, vehicle, quote, monthly_income, employment_type)
        request_id = save_application.invoke(
            {
                "customer": customer,
                "vehicle": vehicle,
                "quote": quote,
                "monthly_income": 50000,
                "employment_type": "salaried",
            }
        )

        print(f"   Result ID: {request_id}")

        if "ERROR" not in request_id:
            print(f"✅ Application saved successfully.")

            print("   Testing check_application_status...")
            status = check_application_status.invoke({"request_id": request_id})
            print(f"   Status Result: {status}")

            if status.get("found"):
                print("✅ Status check passed.")
            else:
                print("❌ Status check failed to find the record.")
        else:
            print(
                f"⚠️ Save returned expected error (if DB not config/schema mismatch): {request_id}"
            )

    except Exception as e:
        print(f"❌ Supabase tool failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_search_tool())
    asyncio.run(test_supabase_tools())
