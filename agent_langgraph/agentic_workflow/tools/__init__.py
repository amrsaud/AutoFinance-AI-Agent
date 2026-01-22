# Copyright 2025 DataRobot, Inc.
# Tools package for AutoFinance AI Agent
"""
External tool integrations for the AutoFinance AI Agent.

Includes:
- Tavily Search for Egyptian car marketplaces
- Supabase for state persistence and application storage
- Installment calculator using PMT formula
- Policy RAG for credit policy retrieval
"""

from tools.installment_calculator import calculate_monthly_installment
from tools.tavily_search import search_vehicles

__all__ = [
    "search_vehicles",
    "calculate_monthly_installment",
]
