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
"""
AutoFinance AI Agent - Financial Co-Pilot for Egypt.
Single-turn workflow: processes one message and responds.
"""

import json
import re
from typing import Any

from datarobot_genai.langgraph.agent import LangGraphAgent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_litellm.chat_models import ChatLiteLLM
from langgraph.graph import END, START, StateGraph

from .config import Config
from .state import AutoFinanceState, CustomerInfo, SearchParams
from .tools.calculator import calculate_installment
from .tools.policy_rag import get_credit_policy
from .tools.supabase_client import check_application_status, save_application
from .tools.tavily_search import search_vehicles

config = Config()


class AutoFinanceAgent(LangGraphAgent):
    """AutoFinance AI Agent - single turn per invocation."""

    @property
    def workflow(self) -> StateGraph[AutoFinanceState]:
        """Build single-turn workflow: router -> handler -> END."""
        wf = StateGraph(AutoFinanceState)

        # Single router node that handles everything
        wf.add_node("router", self.router_node)

        # Simple flow: START -> router -> END
        wf.add_edge(START, "router")
        wf.add_edge("router", END)

        return wf  # type: ignore

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([("user", "{input}")])

    def llm(
        self, preferred_model: str | None = None, auto_model_override: bool = True
    ) -> ChatLiteLLM:
        api_base = self.litellm_api_base(config.llm_deployment_id)
        model = preferred_model or config.llm_default_model
        if auto_model_override and not config.use_datarobot_llm_gateway:
            model = config.llm_default_model
        return ChatLiteLLM(
            model=model,
            api_base=api_base,
            api_key=self.api_key,
            timeout=self.timeout,
            streaming=True,
            max_retries=3,
        )

    async def router_node(
        self, state: AutoFinanceState, config: RunnableConfig = None
    ) -> dict[str, Any]:
        """Route and handle based on current phase and user input."""
        messages = state.get("messages", [])
        phase = state.get("current_phase", "onboarding")

        # Get last user message
        last_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_msg = msg.content.lower()
                break

        if not last_msg:
            return self._welcome_response()

        # Route based on phase and message content
        if phase == "onboarding":
            return await self._handle_onboarding(last_msg, state, config)
        elif phase == "search":
            return await self._handle_search(last_msg, state, config)
        elif phase == "validation":
            return await self._handle_validation(last_msg, state, config)
        elif phase == "selection":
            return await self._handle_selection(last_msg, state, config)
        elif phase == "profiling":
            return await self._handle_profiling(last_msg, state, config)
        elif phase == "quotation":
            return await self._handle_quotation(last_msg, state, config)
        elif phase == "lead_capture":
            return await self._handle_lead_capture(last_msg, state, config)
        else:
            return self._welcome_response()

    def _welcome_response(self) -> dict[str, Any]:
        return {
            "messages": [
                AIMessage(
                    content="""🚗 **Welcome to AutoFinance AI!**

I help you find cars in Egypt and calculate loan options instantly.

**What would you like to do?**
1️⃣ **Start New Request** - Find a vehicle (e.g. "Find me a Toyota Corolla")
2️⃣ **Check Status** - Look up existing application (provide Request ID)"""
                )
            ],
            "current_phase": "onboarding",
        }

    async def _handle_onboarding(
        self, msg: str, state: AutoFinanceState, config: RunnableConfig = None
    ) -> dict[str, Any]:
        # Check for status request
        if any(w in msg for w in ["status", "check", "af-"]):
            match = re.search(r"af-\d{6,}-\d{4}", msg, re.IGNORECASE)
            if match:
                result = check_application_status.invoke(
                    match.group(0).upper(), config=config
                )
                if result.get("found"):
                    return {
                        "messages": [
                            AIMessage(
                                content=f"📋 **Status:** {result['status']}\n**Vehicle:** {result.get('vehicle_name', 'N/A')}"
                            )
                        ],
                        "current_phase": "onboarding",
                    }
                return {
                    "messages": [
                        AIMessage(content=f"❌ No application found: {match.group(0)}")
                    ],
                    "current_phase": "onboarding",
                }
            return {
                "messages": [
                    AIMessage(
                        content="Please provide your Request ID (format: AF-XXXXXX-XXXX)"
                    )
                ],
                "current_phase": "onboarding",
            }

        # Parse car search request
        return await self._parse_search_request(msg)

    async def _parse_search_request(self, msg: str) -> dict[str, Any]:
        """Use LLM to extract search parameters."""
        prompt = f"""Extract car search parameters from: "{msg}"
Return ONLY a JSON object with: make, model, year_from, year_to, price_cap
Example: {{"make":"Toyota","model":"Corolla","year_from":2020,"year_to":2025,"price_cap":null}}
If unclear, use reasonable defaults. year_from should be 5 years ago if not specified."""

        try:
            llm = self.llm()
            response = await llm.ainvoke(prompt)

            # Extract JSON from response
            match = re.search(r"\{[^}]+\}", response.content)
            if match:
                params = json.loads(match.group(0))
                search_params = SearchParams(**params)

                return {
                    "messages": [
                        AIMessage(
                            content=f"""🔍 **Search Parameters**

- **Make:** {search_params.make}
- **Model:** {search_params.model}
- **Years:** {search_params.year_from} - {search_params.year_to}
- **Max Price:** {f"{search_params.price_cap:,.0f} EGP" if search_params.price_cap else "No limit"}

**Is this correct?** (Yes to search, or tell me what to change)"""
                        )
                    ],
                    "search_params": search_params,
                    "current_phase": "validation",
                }
        except Exception as e:
            print(f"Parse error: {e}")

        return {
            "messages": [
                AIMessage(
                    content="I couldn't understand that. Please tell me what car you're looking for.\nExample: 'Find me a 2022 Toyota Corolla under 500,000 EGP'"
                )
            ],
            "current_phase": "onboarding",
        }

    async def _handle_search(
        self, msg: str, state: AutoFinanceState, config: RunnableConfig = None
    ) -> dict[str, Any]:
        # Re-parse if they're giving new criteria
        return await self._parse_search_request(msg)

    async def _handle_validation(
        self, msg: str, state: AutoFinanceState, config: RunnableConfig = None
    ) -> dict[str, Any]:
        if any(w in msg for w in ["yes", "correct", "ok", "search", "proceed"]):
            # Execute search
            params = state.get("search_params")
            if not params:
                return {
                    "messages": [AIMessage(content="What car are you looking for?")],
                    "current_phase": "onboarding",
                }

            # Handle both Pydantic object and dict (serialization safety)
            if hasattr(params, "model_dump"):
                params_dict = params.model_dump()
            elif isinstance(params, dict):
                params_dict = params
            else:
                params_dict = dict(params)

            raw_results = search_vehicles.invoke({"params": params_dict}, config=config)

            if not raw_results:
                return {
                    "messages": [
                        AIMessage(
                            content="No vehicles found. Try different criteria?\nExample: 'Toyota Corolla 2020-2024'"
                        )
                    ],
                    "search_results": [],
                    "current_phase": "onboarding",
                }

            vehicles = await self._parse_vehicles_with_llm(raw_results, params)

            if not vehicles:
                return {
                    "messages": [
                        AIMessage(
                            content="Found results but none matched your exact criteria. Try broadening your search."
                        )
                    ],
                    "search_results": [],
                    "current_phase": "onboarding",
                }

            results_msg = f"📊 **Found {len(vehicles)} vehicles:**\n\n"
            for i, v in enumerate(vehicles, 1):
                results_msg += (
                    f"**{i}. {v.year} {v.make} {v.model}** - {v.price:,.0f} EGP\n"
                )
                mileage_info = f"{v.mileage:,} km" if v.mileage else "N/A"
                if v.source_url:
                    results_msg += (
                        f"   📍 {mileage_info} | [{v.source_name}]({v.source_url})\n"
                    )
                else:
                    results_msg += f"   📍 {mileage_info} | {v.source_name}\n"

            results_msg += f"\n**Which one interests you?** (Enter 1-{len(vehicles)})"

            return {
                "messages": [AIMessage(content=results_msg)],
                "search_results": vehicles,
                "current_phase": "selection",
            }

        # They want to modify
        return await self._parse_search_request(msg)

    async def _parse_vehicles_with_llm(
        self, results: list[dict], params: SearchParams
    ) -> list[Any]:
        """Use agent's LLM to parse raw search results."""
        snippets = []
        for i, res in enumerate(results):
            snippets.append(
                f"Result {i + 1}:\nTitle: {res.get('title')}\nURL: {res.get('url')}\nContent: {res.get('content')}"
            )

        context = "\n\n".join(snippets)
        from .state import Vehicle  # Local import to avoid circular issues if any

        prompt = f"""You are a data extractor for a car finding agent.
Extract vehicle details from the following search results for: {params.make} {params.model} {params.year_from}-{params.year_to}

Rules:
1. Extract strict JSON list of objects.
2. Fields: make, model, year (int), price (float), mileage (int/null), source_name, source_url.
3. Ignore results that are NOT for the specific car requested.
4. Convert price to EGP float (remove commas).
5. Mileage in km.
6. Identify source from URL (Hatla2ee, Dubizzle, OLX, etc).

Search Results:
{context}

Return ONLY valid JSON list:
[
  {{
    "make": "Toyota", "model": "Corolla", "year": 2022, "price": 500000,
    "mileage": 50000, "source_name": "Hatla2ee", "source_url": "..."
  }}
]"""
        try:
            llm = self.llm()
            response = await llm.ainvoke(prompt)
            content = response.content.strip()
            # Clean markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content)
            vehicles = []
            for item in data:
                v = Vehicle(
                    make=item.get("make", params.make),
                    model=item.get("model", params.model),
                    year=item.get("year", params.year_from),
                    price=item.get("price", 0.0),
                    mileage=item.get("mileage"),
                    source_url=item.get("source_url", ""),
                    source_name=item.get("source_name", "Unknown"),
                )
                # Validation
                if v.price > 0:
                    if not params.price_cap or v.price <= params.price_cap:
                        vehicles.append(v)
            return vehicles
        except Exception as e:
            print(f"LLM parsing error: {e}")
            return []

    async def _handle_selection(
        self, msg: str, state: AutoFinanceState, config: RunnableConfig = None
    ) -> dict[str, Any]:
        vehicles = state.get("search_results", [])

        try:
            idx = int(re.search(r"\d+", msg).group(0)) - 1
            if 0 <= idx < len(vehicles):
                v = vehicles[idx]
                return {
                    "messages": [
                        AIMessage(
                            content=f"""✅ **Selected:** {v.year} {v.make} {v.model} - {v.price:,.0f} EGP

💼 Now I need your financial info:
- **Monthly income** (in EGP)
- **Employment type** (Salaried / Self-Employed / Corporate)

Example: "I earn 30,000 EGP and I'm salaried" """
                        )
                    ],
                    "selected_vehicle": v,
                    "current_phase": "profiling",
                }
        except (ValueError, AttributeError):
            pass

        return {
            "messages": [
                AIMessage(content="Please enter a number 1-5 to select a vehicle.")
            ],
            "current_phase": "selection",
        }

    async def _handle_profiling(
        self, msg: str, state: AutoFinanceState, config: RunnableConfig = None
    ) -> dict[str, Any]:
        vehicle = state.get("selected_vehicle")
        if not vehicle:
            return self._welcome_response()

        # Extract income
        income = None
        income_match = re.search(r"(\d+[,\d]*)", msg.replace(",", ""))
        if income_match:
            try:
                income = float(income_match.group(1))
            except ValueError:
                pass

        # Extract employment
        employment = None
        if "salaried" in msg or "salary" in msg:
            employment = "salaried"
        elif "self" in msg or "business" in msg:
            employment = "self_employed"
        elif "corporate" in msg or "company" in msg:
            employment = "corporate"

        if not income or not employment:
            return {
                "messages": [
                    AIMessage(
                        content="Please provide your monthly income and employment type.\nExample: 'My income is 25,000 EGP and I'm salaried'"
                    )
                ],
                "monthly_income": income,
                "employment_type": employment,
                "current_phase": "profiling",
            }

        # Check policy
        policy = get_credit_policy.invoke(
            {
                "vehicle": vehicle,
                "monthly_income": income,
                "employment_type": employment,
            },
            config=config,
        )

        if not policy.eligible:
            return {
                "messages": [
                    AIMessage(
                        content=f"❌ **Unable to proceed**\n\n{policy.rejection_reason}\n\nWould you like to search for a different vehicle?"
                    )
                ],
                "monthly_income": income,
                "employment_type": employment,
                "applicable_policy": policy,
                "current_phase": "onboarding",
            }

        # Calculate quote
        quote = calculate_installment.invoke(
            {
                "principal": vehicle.price,
                "annual_rate": policy.interest_rate,
                "tenure_months": policy.max_tenure_months,
            },
            config=config,
        )

        return {
            "messages": [
                AIMessage(
                    content=f"""💳 **Your Loan Quote**

🚗 **{vehicle.year} {vehicle.make} {vehicle.model}** - {vehicle.price:,.0f} EGP

📊 **Terms:**
- Interest Rate: {quote.interest_rate:.1f}% per annum
- Tenure: {quote.tenure_months} months
- **Monthly Installment: {quote.monthly_installment:,.0f} EGP**
- Total Payment: {quote.total_payment:,.0f} EGP

**Ready to apply?** (Yes to proceed)"""
                )
            ],
            "monthly_income": income,
            "employment_type": employment,
            "applicable_policy": policy,
            "financial_quote": quote,
            "current_phase": "quotation",
        }

    async def _handle_quotation(
        self, msg: str, state: AutoFinanceState, config: RunnableConfig = None
    ) -> dict[str, Any]:
        if any(w in msg for w in ["yes", "proceed", "apply", "submit"]):
            return {
                "messages": [
                    AIMessage(
                        content="""📝 **Almost done!**

Please provide your contact information:
- **Full Name**
- **Email**
- **Phone** (e.g., +201234567890)

Example: "Ahmed Mohamed, ahmed@email.com, +201234567890" """
                    )
                ],
                "current_phase": "lead_capture",
            }

        return self._welcome_response()

    async def _handle_lead_capture(
        self, msg: str, state: AutoFinanceState, config: RunnableConfig = None
    ) -> dict[str, Any]:
        vehicle = state.get("selected_vehicle")
        quote = state.get("financial_quote")

        if not vehicle or not quote:
            return self._welcome_response()

        # Parse contact info
        email = re.search(r"[\w.-]+@[\w.-]+\.\w+", msg)
        phone = re.search(r"\+?\d[\d\s-]{8,}", msg)
        parts = msg.split(",")
        name = parts[0].strip() if parts else "Customer"

        if not email:
            return {
                "messages": [
                    AIMessage(
                        content="Please provide a valid email address.\nFormat: Name, email@example.com, +20XXXXXXXXXX"
                    )
                ],
                "current_phase": "lead_capture",
            }

        customer = CustomerInfo(
            full_name=name,
            email=email.group(0),
            phone=phone.group(0).strip() if phone else "Not provided",
        )

        # Save application
        request_id = save_application.invoke(
            {
                "customer": customer,
                "vehicle": vehicle,
                "quote": quote,
                "monthly_income": state.get("monthly_income", 0),
                "employment_type": state.get("employment_type", ""),
            },
            config=config,
        )

        return {
            "messages": [
                AIMessage(
                    content=f"""🎉 **Application Submitted!**

**Your Request ID: {request_id}**

📋 **Summary:**
- Vehicle: {vehicle.year} {vehicle.make} {vehicle.model}
- Monthly: {quote.monthly_installment:,.0f} EGP

We'll contact you at {customer.email} within 24-48 hours.

Save your Request ID: **{request_id}**

Thank you for choosing AutoFinance! 🚗"""
                )
            ],
            "customer_info": customer,
            "request_id": request_id,
            "current_phase": "onboarding",
        }


MyAgent = AutoFinanceAgent
