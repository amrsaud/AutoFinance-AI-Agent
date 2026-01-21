#!/usr/bin/env python3
"""
Simple CLI tester for the AutoFinance AI Agent.
Tests the LangGraph workflow directly without the DataRobot framework.
"""

import asyncio
import sys
import os

# Add the agentic_workflow directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agentic_workflow"))

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver


async def test_agent(user_input: str):
    """Run the agent with a user input and print results."""
    from agent import AutoFinanceAgent
    from state import AutoFinanceState

    print("\n" + "=" * 60)
    print("🚗 AutoFinance AI Agent - CLI Test")
    print("=" * 60)
    print(f"\n📝 User Input: {user_input}\n")

    # Create agent instance
    agent = AutoFinanceAgent()

    # Build the graph with memory checkpointer for interrupt support
    checkpointer = MemorySaver()
    graph = agent.workflow.compile(checkpointer=checkpointer)

    # Initial state
    initial_state: AutoFinanceState = {
        "messages": [HumanMessage(content=user_input)],
        "current_phase": "welcome",
        "last_action": None,
        "error": None,
        "validation_result": None,
        "search_params": None,
        "search_results": None,
        "selected_vehicle": None,
        "monthly_income": None,
        "employment_type": None,
        "applicable_policy": None,
        "financial_quote": None,
        "risk_profile": None,
        "customer_info": None,
        "request_id": None,
    }

    config = {
        "configurable": {"thread_id": "test-session-1"},
        "recursion_limit": 10,  # Limit to prevent infinite loops during testing
    }

    print("🔄 Starting workflow...\n")
    print("-" * 60)

    try:
        # Stream events from the graph
        async for event in graph.astream(initial_state, config=config):
            for node_name, node_output in event.items():
                print(f"\n📍 Node: {node_name}")
                print("-" * 40)

                # Print messages if present
                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        role = type(msg).__name__.replace("Message", "")
                        print(f"  💬 [{role}] {msg.content[:500]}...")

                # Print other state changes
                for key, value in node_output.items():
                    if key != "messages" and value is not None:
                        print(f"  📊 {key}: {value}")

        print("\n" + "=" * 60)
        print("✅ Workflow completed!")
        print("=" * 60)

        # Check for interrupt
        state = graph.get_state(config)
        if state.next:
            print(f"\n⏸️  Workflow interrupted at: {state.next}")
            print("   Send another input to continue the conversation.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


async def interactive_mode():
    """Run an interactive CLI session."""
    from agent import AutoFinanceAgent
    from state import AutoFinanceState

    print("\n" + "=" * 60)
    print("🚗 AutoFinance AI Agent - Interactive CLI")
    print("=" * 60)
    print("\nType your message and press Enter. Type 'quit' to exit.\n")

    # Create agent instance
    agent = AutoFinanceAgent()

    # Build the graph with memory checkpointer
    checkpointer = MemorySaver()
    graph = agent.workflow.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "interactive-session"}}
    first_message = True

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break

            if not user_input:
                continue

            # Build state based on whether this is first message or continuation
            if first_message:
                state_input: AutoFinanceState = {
                    "messages": [HumanMessage(content=user_input)],
                    "current_phase": "welcome",
                    "last_action": None,
                    "error": None,
                    "validation_result": None,
                    "search_params": None,
                    "search_results": None,
                    "selected_vehicle": None,
                    "monthly_income": None,
                    "employment_type": None,
                    "applicable_policy": None,
                    "financial_quote": None,
                    "risk_profile": None,
                    "customer_info": None,
                    "request_id": None,
                }
                first_message = False
            else:
                # For continuation, just send the new message
                state_input = {"messages": [HumanMessage(content=user_input)]}

            print("\n🤖 Agent:")
            print("-" * 40)

            async for event in graph.astream(state_input, config=config):
                for node_name, node_output in event.items():
                    print(f"\n  📍 [{node_name}]")

                    if "messages" in node_output:
                        for msg in node_output["messages"]:
                            if hasattr(msg, "content") and msg.content:
                                role = type(msg).__name__.replace("Message", "")
                                if role != "Human":
                                    print(f"    {msg.content}")

            # Check state
            state = graph.get_state(config)
            if state.next:
                print(f"\n  ⏸️ Waiting at: {state.next}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single message mode
        user_msg = " ".join(sys.argv[1:])
        asyncio.run(test_agent(user_msg))
    else:
        # Interactive mode
        asyncio.run(interactive_mode())
