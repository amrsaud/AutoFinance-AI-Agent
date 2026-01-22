# Copyright 2025 DataRobot, Inc.
# LangGraph nodes package for AutoFinance AI Agent
"""
LangGraph node implementations for the AutoFinance workflow.

Nodes:
- router: Entry point with conditional routing
- search_param: LLM-powered parameter extraction
- market_search: Tavily API integration
- profiling_logic: Data collection loop
- ask_questions: Income/employment prompts
- policy_rag: Credit policy retrieval
- quotation: Loan calculation
- submission: Supabase storage
- status_check: Query existing applications
"""

from nodes.ask_questions import ask_employment_node, ask_income_node
from nodes.market_search import market_search_node
from nodes.policy_rag import policy_rag_node
from nodes.profiling_logic import profiling_logic_node, route_profiling
from nodes.quotation import quotation_node
from nodes.router import route_initial, router_node
from nodes.search_param import search_param_node
from nodes.status_check import status_check_node
from nodes.submission import submission_node

__all__ = [
    "router_node",
    "route_initial",
    "search_param_node",
    "market_search_node",
    "profiling_logic_node",
    "route_profiling",
    "ask_income_node",
    "ask_employment_node",
    "policy_rag_node",
    "quotation_node",
    "submission_node",
    "status_check_node",
]
