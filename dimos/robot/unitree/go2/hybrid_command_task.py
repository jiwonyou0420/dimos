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

"""Small LLM-backed command router for the hackathon hybrid stack.

This deliberately avoids the full LangGraph agent path. It only translates
flexible language into a validated macro action, then calls the same reliable
DimOS skills as the deterministic voice layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re

from dimos.agents.annotation import skill
from dimos.core.module import Module

_PLACEHOLDER_API_KEYS = {"", "false", "none", "null", "changeme", "todo", "your-key-here"}

_SYSTEM_PROMPT = """
You route flexible natural-language requests for a Unitree Go2 demo.

Return JSON only with:
  action: one of find_object, navigate_with_text, explore, query_memory,
          lock_person, follow_person, stop, status, clarify, reject
  text: short object/place/query/action text, or empty string
  response: short spoken response

Use find_object for visible object search, empty-chair search, remembered-object
queries, person lock/follow commands, and stop commands when applicable.
Use navigate_with_text for open-vocabulary places such as kitchen, doorway,
hallway, entrance, desk area, or semantic-map locations.
Use clarify or reject for unsafe or unclear requests. Never invent tools.
"""


@dataclass(frozen=True)
class HybridRoute:
    action: str
    text: str = ""
    response: str = ""


class HybridCommandTask(Module):
    """Translate flexible commands into safe macro skills."""

    rpc_calls: list[str] = [
        "FindObjectTask.find_object",
        "FindObjectTask.find_object_status",
        "FindObjectTask.semantic_memory_status",
        "NavigationSkillContainer.navigate_with_text",
        "NavigationSkillContainer.stop_navigation",
    ]

    @skill
    def hybrid_command(self, command: str) -> str:
        """Route a flexible language command through approved macro skills.

        Args:
            command: Natural-language user command that did not map cleanly to
                a deterministic hardcoded action.
        """

        cleaned = _normalize(command)
        if not cleaned:
            return "I need a command."

        route = self._route_with_llm(cleaned) or _route_with_rules(cleaned)
        return self._execute_route(route, cleaned)

    def _route_with_llm(self, command: str) -> HybridRoute | None:
        if not _has_llm_api_key():
            return None

        try:
            from openai import OpenAI

            base_url = os.getenv("GO2_VOICE_BASE_URL") or os.getenv("OPENAI_BASE_URL")
            client = OpenAI(base_url=base_url) if base_url else OpenAI()
            response = client.chat.completions.create(
                model=os.getenv("GO2_AGENT_MODEL") or os.getenv("GO2_VOICE_MODEL") or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": command},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
                timeout=8.0,
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            action = str(data.get("action", "")).strip().lower()
            text = _normalize(str(data.get("text", "") or ""))
            response_text = str(data.get("response", "") or "").strip()
            if action in _ALLOWED_ACTIONS:
                return HybridRoute(action=action, text=text, response=response_text)
        except Exception:
            return None
        return None

    def _execute_route(self, route: HybridRoute, original_command: str) -> str:
        action = route.action
        text = route.text

        if action == "find_object":
            return str(self.get_rpc_calls("FindObjectTask.find_object")(_find_object_command(text)))
        if action == "explore":
            return str(self.get_rpc_calls("FindObjectTask.find_object")("explore"))
        if action == "query_memory":
            target = text or _memory_target_from_text(original_command)
            return str(self.get_rpc_calls("FindObjectTask.find_object")(f"where is my {target}"))
        if action == "lock_person":
            return str(self.get_rpc_calls("FindObjectTask.find_object")("can you see me"))
        if action == "follow_person":
            return str(self.get_rpc_calls("FindObjectTask.find_object")("follow me"))
        if action == "stop":
            return str(self.get_rpc_calls("FindObjectTask.find_object")("stop"))
        if action == "navigate_with_text":
            query = text or original_command
            return str(self.get_rpc_calls("NavigationSkillContainer.navigate_with_text")(query))
        if action == "status":
            return str(self.get_rpc_calls("FindObjectTask.find_object_status")())
        if action == "reject":
            return route.response or "I cannot do that safely."
        if action == "clarify":
            return route.response or "Please give me a clearer target."

        return "I could not route that command."


_ALLOWED_ACTIONS = {
    "find_object",
    "navigate_with_text",
    "explore",
    "query_memory",
    "lock_person",
    "follow_person",
    "stop",
    "status",
    "clarify",
    "reject",
}


def _route_with_rules(command: str) -> HybridRoute:
    tokens = set(re.findall(r"[a-z0-9]+", command))

    if _is_smalltalk_or_general_question(command, tokens):
        return HybridRoute("clarify", response="I am ready for a robot command.")
    if tokens & {"stop", "halt", "cancel", "freeze"}:
        return HybridRoute("stop")
    if "follow" in tokens and ("me" in tokens or "person" in tokens):
        return HybridRoute("follow_person")
    if {"see", "me"} <= tokens or {"look", "at", "me"} <= tokens:
        return HybridRoute("lock_person")
    if tokens & {"explore", "scan", "remember"}:
        return HybridRoute("explore")
    if tokens & {"where", "lost", "forgot", "remember"} and tokens & {"bag", "backpack"}:
        return HybridRoute("query_memory", "bag")
    if tokens & {"sit", "seat", "seating"}:
        return HybridRoute("find_object", "empty chair")
    if tokens & {"drink", "drinking"}:
        return HybridRoute("find_object", "bottle")
    if tokens & {"kitchen", "door", "doorway", "hall", "hallway", "entrance", "exit", "room"}:
        return HybridRoute("navigate_with_text", command)

    return HybridRoute("clarify", response="Please give me a robot command with a target.")


def _is_smalltalk_or_general_question(command: str, tokens: set[str]) -> bool:
    if command in {
        "how are you",
        "how are you doing",
        "how is it going",
        "hows it going",
        "what is up",
        "whats up",
        "thank you",
        "thanks",
        "good job",
    }:
        return True
    if tokens & {"you", "yourself"} and tokens & {"how", "what", "who", "why"}:
        return True
    return False


def _normalize(text: str) -> str:
    compact = re.sub(r"\s+", " ", (text or "").strip().lower())
    return compact.strip(" .!?")


def _has_llm_api_key() -> bool:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return len(key) >= 20 and key.lower() not in _PLACEHOLDER_API_KEYS


def _find_object_command(text: str) -> str:
    target = text or "object"
    if target.startswith(("find ", "where is ", "can you see", "follow ")):
        return target
    return f"find {target}"


def _memory_target_from_text(text: str) -> str:
    if "backpack" in text or "bag" in text:
        return "bag"
    return "object"


hybrid_command_task = HybridCommandTask.blueprint

__all__ = ["HybridCommandTask", "hybrid_command_task"]
