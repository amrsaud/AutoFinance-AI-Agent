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
PostgreSQL Checkpointer using official langgraph-checkpoint-postgres.

Uses the official LangGraph PostgresSaver for state persistence with Supabase.
Falls back to InMemorySaver if Postgres connection fails.
"""

from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

from config import Config


def create_postgres_checkpointer() -> Optional[BaseCheckpointSaver]:
    """
    Create a PostgresSaver checkpointer for Supabase.

    Uses the official langgraph-checkpoint-postgres package which:
    - Auto-creates required tables via .setup()
    - Handles all serialization/deserialization
    - Supports both sync and async operations

    Falls back to InMemorySaver if Postgres connection fails.

    Returns:
        PostgresSaver if configured, MemorySaver as fallback
    """
    config = Config()

    postgres_uri = config.postgres_uri

    if not postgres_uri:
        print("⚠️  POSTGRES_URI not configured, using in-memory checkpointer")
        print("   Memory will work but won't persist across server restarts")
        return MemorySaver()

    try:
        # Create connection and checkpointer
        import psycopg
        from psycopg.rows import dict_row

        print("🔌 Connecting to PostgreSQL...")

        conn = psycopg.connect(
            postgres_uri,
            autocommit=True,
            row_factory=dict_row,
            connect_timeout=10,
        )

        checkpointer = PostgresSaver(conn)

        # Setup creates required tables if they don't exist
        checkpointer.setup()

        print("✅ PostgreSQL checkpointer initialized successfully")
        return checkpointer

    except Exception as e:
        print(f"⚠️  PostgreSQL connection failed: {e}")
        print("   Falling back to in-memory checkpointer")
        print("   Memory will work but won't persist across server restarts")
        print("")
        print("   To fix: Update POSTGRES_URI in .env with correct format:")
        print(
            "   postgres://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.co:6543/postgres"
        )
        return MemorySaver()


# For backwards compatibility
SupabaseCheckpointer = PostgresSaver
create_supabase_checkpointer = create_postgres_checkpointer
