#!/usr/bin/env python3

# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hybrid hackathon blueprint.

This stack keeps the deterministic find-object task as the preferred execution
path and adds a small LLM-backed macro router for commands that the app layer
cannot map confidently to a hardcoded skill.
"""

from __future__ import annotations

from dimos.agents.mcp.mcp_server import McpServer
from dimos.agents.skills.navigation import navigation_skill
from dimos.core.blueprints import autoconnect
from dimos.perception.spatial_perception import spatial_memory
from dimos.robot.unitree.go2.blueprints.smart.unitree_go2 import unitree_go2
from dimos.robot.unitree.go2.find_object_task import find_object_task
from dimos.robot.unitree.go2.hybrid_command_task import hybrid_command_task

unitree_go2_hackathon_hybrid = autoconnect(
    unitree_go2,
    spatial_memory(),
    find_object_task(),
    hybrid_command_task(),
    navigation_skill(),
    McpServer.blueprint(),
).global_config(n_workers=9, robot_model="unitree_go2", planner_robot_speed=0.15)

__all__ = ["unitree_go2_hackathon_hybrid"]
