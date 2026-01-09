#!/usr/bin/env python
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
import logging
import os
import shutil
from pathlib import Path
from typing import Union

logger = logging.getLogger()

work_dir = Path(os.path.dirname(__file__))


def try_to_remove(path: Union[str, Path]):
    """Attempt to remove a file or directory, ignoring errors."""
    try:
        if os.path.isdir(str(path)):
            shutil.rmtree(str(path), ignore_errors=True)
        else:
            os.remove(str(path))
    except Exception as e:
        logger.info(f"Warning: Could not remove {path}: {e}")


def remove_agent_environment(agent_name: str):
    """Remove the agent environment if it exists."""
    agent_env_path = work_dir / f"{agent_name}"
    if agent_env_path.exists():
        logger.info(f"Removing existing agent environment: {agent_env_path}")
        try_to_remove(str(agent_env_path))
        try_to_remove(
            str(work_dir / ".github" / "workflows" / f"{agent_name}-test.yml")
        )
        try_to_remove(
            str(work_dir / ".datarobot" / "answers" / f"agent-{agent_name}.yml")
        )
        try_to_remove(str(work_dir / ".datarobot" / "cli" / f"{agent_name}.yaml"))
        try_to_remove(str(work_dir / "infra" / "feature_flags" / f"{agent_name}.yaml"))
        try_to_remove(str(work_dir / "infra" / "infra" / f"{agent_name}.py"))
        try_to_remove(str(work_dir / f"Taskfile_{agent_name}.yml"))
        logger.info(f"Removed agent environment: {agent_env_path}")
    else:
        print(f"No existing agent environment found at: {agent_env_path}")


def main():
    print("    ____        __        ____        __          __  ")
    print("   / __ \____ _/ /_____ _/ __ \____  / /_  ____  / /_ ")
    print("  / / / / __ `/ __/ __ `/ /_/ / __ \/ __ \/ __ \/ __/ ")
    print(" / /_/ / /_/ / /_/ /_/ / _, _/ /_/ / /_/ / /_/ / /_   ")
    print("/_____/\__,_/\__/\__,_/_/ |_|\____/_.___/\____/\__/   ")
    print()
    print("-------------------------------------------------------")
    print("          Quickstart for DataRobot AI Agents           ")
    print("-------------------------------------------------------")

    agent_templates = [
        "agent_crewai",
        "agent_generic_base",
        "agent_langgraph",
        "agent_llamaindex",
        "agent_nat",
    ]
    print("\nYou will now select an agentic framework to use for this project.")
    print("For more information on the different agentic frameworks please go to:")
    print(
        "  https://github.com/datarobot-community/datarobot-agent-templates/blob/main/docs/getting-started.md"
    )
    print()
    print("Please select an agentic framework to use:")
    for i, template in enumerate(agent_templates, start=1):
        print(f"{i}. {template}")
    choice = input("Enter your choice (1-5): ")
    if choice not in ["1", "2", "3", "4", "5"]:
        print("Invalid choice. Exiting.")
        return
    else:
        template_name = agent_templates[int(choice) - 1]
        print(f"You selected: {template_name}")
        print("Setting up the agent environment...")
        print("Cleaning up other framework templates to streamline your workspace.")
        agent_templates_to_remove = [
            agent for agent in agent_templates if agent != template_name
        ]
        for agent in agent_templates_to_remove:
            remove_agent_environment(agent)


if __name__ == "__main__":
    main()
