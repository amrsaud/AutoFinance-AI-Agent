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
Policy RAG Node - Credit policy retrieval and eligibility check.

Queries the vector database (or fallback policies) to retrieve
applicable credit terms based on user profile and vehicle.
"""

from typing import Any

from config import Config
from langchain_core.messages import AIMessage
from models import AgentState, WorkflowPhase
from tools.policy_rag import format_policy_summary, retrieve_credit_policy

config = Config()


def policy_rag_node(state: AgentState) -> dict[str, Any]:
    """
    Retrieve credit policy and check eligibility.

    Uses DataRobot VectorDB when configured, otherwise falls back
    to predefined policy rules.
    """
    # Validate required data
    if not state.selected_vehicle:
        return {
            "messages": [
                AIMessage(
                    content="⚠️ No vehicle selected. Please select a vehicle first."
                )
            ],
            "current_phase": WorkflowPhase.DISCOVERY,
        }

    if state.monthly_income is None:
        return {
            "messages": [
                AIMessage(
                    content="⚠️ Income not provided. Please provide your monthly income."
                )
            ],
        }

    if state.employment_type is None:
        return {
            "messages": [
                AIMessage(
                    content="⚠️ Employment type not provided. Please specify your employment type."
                )
            ],
        }

    # Retrieve applicable policy
    policy = retrieve_credit_policy(
        employment_type=state.employment_type,
        monthly_income=state.monthly_income,
        vehicle=state.selected_vehicle,
        vectordb_id=config.vectordb_id,
    )

    # Format response based on eligibility
    policy_message = format_policy_summary(policy)

    if policy.is_eligible:
        # Proceed to quotation phase
        return {
            "messages": [AIMessage(content=policy_message)],
            "applicable_policy": policy,
            "current_phase": WorkflowPhase.QUOTATION,
        }
    else:
        # Not eligible - offer options
        return {
            "messages": [AIMessage(content=policy_message)],
            "applicable_policy": policy,
            "current_phase": WorkflowPhase.DISCOVERY,  # Return to discovery to try again
        }
