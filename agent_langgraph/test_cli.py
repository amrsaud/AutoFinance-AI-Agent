# Copyright 2025 DataRobot, Inc.
# Licensed under the Apache License, Version 2.0
"""
CLI Test script for AutoFinance AI Agent.
Simulates a multi-turn conversation to test the agent flow.
"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from agentic_workflow.agent import AutoFinanceAgent
from agentic_workflow.config import Config

config = Config()


async def run_cli_test():
    """Run a simulated conversation from the CLI."""
    print("🚗 AutoFinance AI Agent - CLI Test Mode")
    print("---------------------------------------")

    # Initial state
    state = {
        "messages": [],
        "current_phase": "onboarding",
        "search_params": None,
        "search_results": None,
        "selected_vehicle": None,
        "monthly_income": None,
        "employment_type": None,
        "applicable_policy": None,
        "financial_quote": None,
        "customer_info": None,
        "request_id": None,
    }

    # Simulation steps
    # Each step is a user input
    steps = [
        "Find me a 2022 Toyota Corolla",
        "yes",
        "1",  # Select first vehicle
        "My income is 30000 and I'm salaried",
        "yes",  # Accept quote
        "Ahmed, ahmed@email.com, +201234567890",  # Submit
    ]

    for i, user_input in enumerate(steps, 1):
        print(f"\n[Step {i}] User: {user_input}")

        # Add user message
        state["messages"].append(HumanMessage(content=user_input))

        # Create agent
        agent = AutoFinanceAgent(
            messages=state["messages"],
            model=config.llm_default_model,
            stream=False,
        )
        graph = agent.workflow.compile()

        try:
            # Run workflow
            print("... Processing ...")
            result = await graph.ainvoke(state)

            print(f"DEBUG: Result keys: {result.keys()}")
            print(f"DEBUG: Result phase: {result.get('current_phase')}")

            # Update state
            for key in state:
                if key in result and result[key] is not None:
                    state[key] = result[key]

            # Extract and print response
            response_text = ""
            if "messages" in result:
                for msg in result["messages"]:
                    if isinstance(msg, AIMessage) and msg.content:
                        if not msg.content.strip().startswith("{"):
                            response_text = msg.content
                            # Add to state messages if not already last message (graph might append it)
                            # But our simple router loop creates a new AIMessage object each time probably
                            pass

            if response_text:
                print(f"[Agent]:\n{response_text}")
                state["messages"].append(AIMessage(content=response_text))
            else:
                print("[Agent]: (No text response)")

            print(f"Current Phase: {state.get('current_phase')}")

            # Exit if submission complete
            if state.get("request_id"):
                print(f"\n✅ SUCCESS! Request ID: {state['request_id']}")
                break

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()
            break


if __name__ == "__main__":
    asyncio.run(run_cli_test())
