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

"""First-pass object navigation task for the Go2 hackathon flow.

Stage 1 proved command ingress and deterministic parsing:

    "go to the red can" -> target_text="red can"

Stage 2 adds on-demand 2D detection from the current color frame.
Stage 3 estimates a world-frame target position from map points projected into
the bbox. Later stages can call `NavigationInterface.set_goal(...)`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import re
from threading import Event, RLock, Thread, current_thread
import time
from typing import Any

from reactivex.disposable import Disposable

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs import PoseStamped, Quaternion, Twist, Vector3
from dimos.msgs.nav_msgs import OccupancyGrid
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues
from dimos.msgs.sensor_msgs import CameraInfo, Image, ImageFormat, PointCloud2
from dimos.msgs.visualization_msgs.EntityMarkers import EntityMarkers, Marker
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


_STOP_COMMANDS = {
    "stop",
    "halt",
    "cancel",
    "cancel navigation",
    "stop navigation",
    "stop moving",
    "freeze",
}

_FIND_PREFIXES = (
    "go to",
    "goto",
    "navigate to",
    "walk to",
    "move to",
    "approach",
    "find",
    "find me",
    "look for",
    "search for",
    "take me to",
)

_POLICY_IDLE = "idle"
_POLICY_SEARCHING = "searching"
_POLICY_APPROACHING = "approaching"

_POLICY_LOOP_INTERVAL_S = 0.05
_DETECTION_INTERVAL_S = 0.25
_MARKER_INTERVAL_S = 0.1
_SEARCH_TWIST_INTERVAL_S = 0.1
_SEARCH_LINEAR_X_M_S = 0.075
_SEARCH_BLOCKED_LINEAR_X_M_S = -0.04
_SEARCH_YAW_RATE_RAD_S = 0.4
_SEARCH_BLOCKED_OCCUPANCY_THRESHOLD = 65
_GOAL_UPDATE_MIN_INTERVAL_S = 0.8
_GOAL_UPDATE_DISTANCE_M = 0.2
_GOAL_UPDATE_YAW_RAD = math.radians(12)
_TRACK_REASSOCIATION_DISTANCE_M = 0.9
_TRACK_SMOOTHING_ALPHA = 0.25
_MAX_CONSISTENCY_CANDIDATES = 6
_POSE_CLUSTER_CELL_M = 0.25
_POSE_CLUSTER_MIN_POINTS = 20
_POSE_CLUSTER_MIN_RELATIVE_TO_LARGEST = 0.2
_MANUAL_TWIST_INTERVAL_S = 0.1
_MANUAL_FORWARD_DISTANCE_M = 1.0
_MANUAL_FORWARD_SPEED_M_S = 0.15
_MANUAL_TURN_ANGLE_RAD = math.radians(35.0)
_MANUAL_TURN_YAW_RATE_RAD_S = 0.5


@dataclass(frozen=True)
class FindObjectPlan:
    intent: str
    raw_text: str
    target_text: str | None = None
    accepted: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class FindObjectDetection:
    target_text: str
    detected_label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    center_xy: tuple[float, float]
    area_px: float
    image_width: int
    image_height: int
    matching_count: int
    detected_labels: list[str]
    world_xyz: tuple[float, float, float] | None = None
    world_frame_id: str | None = None
    pose_method: str | None = None
    object_point_count: int | None = None
    pose_cluster_count: int | None = None
    pose_ambiguity_handled: bool = False
    robot_xyz: tuple[float, float, float] | None = None
    nav_goal_xyz: tuple[float, float, float] | None = None
    nav_goal_yaw: float | None = None
    standoff_m: float | None = None
    navigation_started: bool = False
    navigation_error: str | None = None
    segmentation_used: bool = False


class FindObjectTask(Module):
    """Small task module for text-commanded object navigation.

    Current behavior:
    - subscribes to `/human_input` text when present;
    - exposes `find_object(command)` as an MCP/agent skill;
    - parses the requested target;
    - starts a small FSM that searches, tracks, and approaches the object;
    - keeps running 2D detection on the latest color image while active;
    - estimates a world-frame target from map points inside the segmentation mask/bbox;
    - updates a buffered navigation goal that faces the object;
    - publishes a marked debug image on `detected_image`;
    - records/publishes status;
    - implements stop by calling `NavigationInterface.cancel_goal` when available.
    """

    rpc_calls: list[str] = [
        "NavigationInterface.set_goal",
        "NavigationInterface.is_goal_reached",
        "NavigationInterface.cancel_goal",
    ]

    human_input: In[str]
    color_image: In[Image]
    camera_info: In[CameraInfo]
    global_map: In[PointCloud2]
    global_costmap: In[OccupancyGrid]
    odom: In[PoseStamped]
    status: Out[str]
    detected_image: Out[Image]
    object_markers: Out[EntityMarkers]
    cmd_vel: Out[Twist]

    def __init__(self) -> None:
        super().__init__()
        self._lock = RLock()
        self._last_plan: FindObjectPlan | None = None
        self._last_detection: FindObjectDetection | None = None
        self._last_message: str = "No command received yet."
        self._latest_image: Image | None = None
        self._latest_camera_info: CameraInfo | None = None
        self._latest_global_map: PointCloud2 | None = None
        self._latest_global_costmap: OccupancyGrid | None = None
        self._latest_odom: PoseStamped | None = None
        self._detector: Any | None = None
        self._detector_mode: str = "uninitialized"
        self._policy_stop_event = Event()
        self._policy_thread: Thread | None = None
        self._manual_motion_stop_event = Event()
        self._manual_motion_thread: Thread | None = None
        self._manual_motion_name: str | None = None
        self._policy_mode = _POLICY_IDLE
        self._policy_target_text: str | None = None
        self._policy_state_changed_ts = time.time()
        self._last_detection_attempt_monotonic = 0.0
        self._last_detection_seen_ts: float | None = None
        self._last_marker_publish_monotonic = 0.0
        self._last_search_twist_monotonic = 0.0
        self._last_goal_sent_monotonic = 0.0
        self._last_sent_goal_xyz: tuple[float, float, float] | None = None
        self._last_sent_goal_yaw: float | None = None
        self._tracked_world_xyz: tuple[float, float, float] | None = None
        self._search_forward_blocked = False
        self._goal_update_count = 0
        self._detection_count = 0

    @rpc
    def start(self) -> None:
        super().start()

        if self.human_input.transport is not None:
            self._disposables.add(Disposable(self.human_input.subscribe(self._on_human_input)))
        if self.color_image.transport is not None:
            self._disposables.add(Disposable(self.color_image.subscribe(self._on_color_image)))
        if self.camera_info.transport is not None:
            self._disposables.add(Disposable(self.camera_info.subscribe(self._on_camera_info)))
        if self.global_map.transport is not None:
            self._disposables.add(Disposable(self.global_map.subscribe(self._on_global_map)))
        if self.global_costmap.transport is not None:
            self._disposables.add(
                Disposable(self.global_costmap.subscribe(self._on_global_costmap))
            )
        if self.odom.transport is not None:
            self._disposables.add(Disposable(self.odom.subscribe(self._on_odom)))

        self._policy_stop_event.clear()
        self._policy_thread = Thread(target=self._policy_loop, daemon=True)
        self._policy_thread.start()

    @rpc
    def stop(self) -> None:
        self._stop_manual_motion(join=True)
        self._transition_to_idle("module stopping", cancel_navigation=False, clear_target=True)
        self._publish_zero_twist()
        self._policy_stop_event.set()
        if self._policy_thread is not None and self._policy_thread is not current_thread():
            self._policy_thread.join(timeout=2.0)
            self._policy_thread = None
        if self._detector is not None and hasattr(self._detector, "stop"):
            self._detector.stop()
        super().stop()

    def _on_human_input(self, text: str) -> None:
        message = self._handle_command(text, source="human_input")
        logger.info(message)

    def _on_color_image(self, image: Image) -> None:
        with self._lock:
            self._latest_image = image

    def _on_camera_info(self, camera_info: CameraInfo) -> None:
        with self._lock:
            self._latest_camera_info = camera_info

    def _on_global_map(self, pointcloud: PointCloud2) -> None:
        with self._lock:
            self._latest_global_map = pointcloud

    def _on_global_costmap(self, costmap: OccupancyGrid) -> None:
        with self._lock:
            self._latest_global_costmap = costmap

    def _on_odom(self, odom: PoseStamped) -> None:
        with self._lock:
            self._latest_odom = odom

    @skill
    def find_object(self, command: str) -> str:
        """Start the object-search-and-approach policy.

        Accepts free text such as "go to the can", "find a red soda can", or
        just "red can", extracts the target object phrase, rotates in place
        while searching, then keeps detecting and updating the navigation goal
        while approaching the selected object.

        Args:
            command: Natural-language object navigation command.
        """

        return self._handle_command(command, source="skill")

    @skill
    def find_object_status(self) -> str:
        """Return the most recent find-object command status."""

        if self._last_plan is None:
            return self._last_message
        return json.dumps(
            {
                "message": self._last_message,
                "detection": asdict(self._last_detection) if self._last_detection else None,
                "plan": asdict(self._last_plan),
                "policy": self._policy_status_snapshot(),
            },
            indent=2,
            sort_keys=True,
        )

    @skill
    def turn_left(self) -> str:
        """Turn the robot left by roughly 90 degrees."""

        return self._start_manual_motion(
            name="turn_left",
            linear_x=0.0,
            angular_z=_MANUAL_TURN_YAW_RATE_RAD_S,
            duration_s=_MANUAL_TURN_ANGLE_RAD / _MANUAL_TURN_YAW_RATE_RAD_S,
            message="Turning left.",
        )

    @skill
    def turn_right(self) -> str:
        """Turn the robot right by roughly 90 degrees."""

        return self._start_manual_motion(
            name="turn_right",
            linear_x=0.0,
            angular_z=-_MANUAL_TURN_YAW_RATE_RAD_S,
            duration_s=_MANUAL_TURN_ANGLE_RAD / _MANUAL_TURN_YAW_RATE_RAD_S,
            message="Turning right.",
        )

    @skill
    def go_forward(self) -> str:
        """Move the robot forward by roughly one meter."""

        return self._start_manual_motion(
            name="go_forward",
            linear_x=_MANUAL_FORWARD_SPEED_M_S,
            angular_z=0.0,
            duration_s=_MANUAL_FORWARD_DISTANCE_M / _MANUAL_FORWARD_SPEED_M_S,
            message="Moving forward.",
        )

    @skill
    def stop_robot(self) -> str:
        """Stop manual motion, object search, and active navigation."""

        return self._execute_stop()

    def _handle_command(self, text: str, *, source: str) -> str:
        plan = parse_find_object_command(text)
        self._last_plan = plan

        if plan.intent == "stop":
            message = self._execute_stop()
        elif plan.intent == "find_object" and plan.target_text:
            message = self._start_find_object_policy(plan.target_text)
        else:
            self._last_detection = None
            message = (
                "I could not parse that as an object navigation command. "
                "Try: 'go to the can' or 'find the red chair'."
            )

        self._last_message = f"[{source}] {message}"
        self.status.publish(
            json.dumps(
                {
                    "ts": time.time(),
                    "source": source,
                    "message": message,
                    "detection": asdict(self._last_detection) if self._last_detection else None,
                    "plan": asdict(plan),
                    "policy": self._policy_status_snapshot(),
                }
            )
        )
        return self._last_message

    def _start_find_object_policy(self, target_text: str) -> str:
        with self._lock:
            self._policy_mode = _POLICY_SEARCHING
            self._policy_target_text = target_text
            self._policy_state_changed_ts = time.time()
            self._tracked_world_xyz = None
            self._last_detection = None
            self._last_detection_seen_ts = None
            self._last_detection_attempt_monotonic = 0.0
            self._last_goal_sent_monotonic = 0.0
            self._last_sent_goal_xyz = None
            self._last_sent_goal_yaw = None
            self._goal_update_count = 0
            self._detection_count = 0

        self._cancel_navigation()
        self._publish_status("searching", f"Searching for '{target_text}'.")
        self._publish_mode_only_image()
        return (
            f"Started object-search policy for '{target_text}'. "
            "I will rotate in place until I see it, then approach with a stand-off goal."
        )

    def _policy_loop(self) -> None:
        while not self._policy_stop_event.is_set():
            try:
                self._policy_tick()
            except Exception:
                logger.warning("Find-object policy tick failed", exc_info=True)
            self._policy_stop_event.wait(_POLICY_LOOP_INTERVAL_S)

    def _policy_tick(self) -> None:
        with self._lock:
            mode = self._policy_mode
            target_text = self._policy_target_text

        if mode == _POLICY_IDLE or target_text is None:
            return

        now = time.monotonic()

        if mode == _POLICY_SEARCHING and now - self._last_search_twist_monotonic >= _SEARCH_TWIST_INTERVAL_S:
            self._publish_search_twist()
            self._last_search_twist_monotonic = now

        if mode == _POLICY_APPROACHING and self._navigation_goal_reached():
            self._transition_to_idle(
                f"Reached '{target_text}'. Returning to idle.",
                cancel_navigation=False,
                clear_target=False,
            )
            self._publish_status("arrived", f"Reached '{target_text}'.")
            return

        if now - self._last_detection_attempt_monotonic >= _DETECTION_INTERVAL_S:
            self._last_detection_attempt_monotonic = now
            self._run_detection_policy_tick(target_text)

        if now - self._last_marker_publish_monotonic >= _MARKER_INTERVAL_S:
            self._publish_active_markers()
            self._last_marker_publish_monotonic = now

    def _run_detection_policy_tick(self, target_text: str) -> None:
        with self._lock:
            tracked_world_xyz = self._tracked_world_xyz
            mode = self._policy_mode

        detection_attempt = self._detect_target(
            target_text,
            tracked_world_xyz=tracked_world_xyz if mode == _POLICY_APPROACHING else None,
        )
        result = detection_attempt["result"]
        detection = detection_attempt["detection"]
        image = detection_attempt["image"]

        if result is None or detection is None or image is None:
            if mode == _POLICY_SEARCHING and image is not None:
                self._publish_unmatched_image(
                    image,
                    target_text,
                    detection_attempt["detected_labels"],
                )
            return

        tracked_result = self._update_tracked_detection(result)
        planned_result = self._prepare_navigation_goal(tracked_result)

        with self._lock:
            if self._policy_mode == _POLICY_IDLE or self._policy_target_text != target_text:
                return

        should_send_goal = self._should_send_goal_update(
            planned_result,
            force=mode == _POLICY_SEARCHING,
        )
        if should_send_goal:
            planned_result = self._send_navigation_goal(planned_result)
        elif self._last_sent_goal_xyz is not None and not planned_result.navigation_error:
            planned_result = replace(planned_result, navigation_started=True)

        with self._lock:
            previous_mode = self._policy_mode
            if previous_mode == _POLICY_IDLE or self._policy_target_text != target_text:
                return
            self._last_detection = planned_result
            self._last_detection_seen_ts = time.time()
            self._detection_count += 1
            if previous_mode == _POLICY_SEARCHING and planned_result.navigation_started:
                self._policy_mode = _POLICY_APPROACHING
                self._policy_state_changed_ts = time.time()

        if previous_mode == _POLICY_SEARCHING and planned_result.navigation_started:
            self._publish_zero_twist()
            self._publish_status(
                "approaching",
                f"Detected '{planned_result.detected_label}' and started approaching.",
            )

        self._publish_object_markers(planned_result)
        self._publish_annotated_image(image, detection, planned_result)

    def _detect_target(
        self,
        target_text: str,
        *,
        tracked_world_xyz: tuple[float, float, float] | None,
    ) -> dict[str, Any]:
        with self._lock:
            image = self._latest_image
            robot_pose = self._latest_odom

        if image is None:
            return {
                "result": None,
                "detection": None,
                "image": None,
                "detected_labels": [],
            }

        detector = self._get_detector()
        image_detections = detector.process_image(image)
        detections = list(image_detections)
        detected_labels = sorted({detection.name for detection in detections})
        matches = [
            detection
            for detection in detections
            if _target_matches_detection(target_text, detection.name)
        ]

        if not matches:
            return {
                "result": None,
                "detection": None,
                "image": image,
                "detected_labels": detected_labels,
            }

        chosen, pose_estimate = self._choose_consistent_detection(
            target_text,
            image,
            matches,
            tracked_world_xyz=tracked_world_xyz,
        )
        if chosen is None:
            return {
                "result": None,
                "detection": None,
                "image": image,
                "detected_labels": detected_labels,
            }

        result = _detection_to_result(
            target_text=target_text,
            detection=chosen,
            image=image,
            matching_count=len(matches),
            detected_labels=detected_labels,
            pose_estimate=pose_estimate,
            robot_pose=robot_pose,
        )
        return {
            "result": result,
            "detection": chosen,
            "image": image,
            "detected_labels": detected_labels,
        }

    def _choose_consistent_detection(
        self,
        target_text: str,
        image: Image,
        matches: list[Any],
        *,
        tracked_world_xyz: tuple[float, float, float] | None,
    ) -> tuple[Any | None, dict[str, Any]]:
        if tracked_world_xyz is None:
            chosen = max(matches, key=_detection_area_px)
            return chosen, self._estimate_detection_pose(chosen, image)

        ranked_matches = sorted(matches, key=_detection_area_px, reverse=True)[
            :_MAX_CONSISTENCY_CANDIDATES
        ]
        best: tuple[float, Any, dict[str, Any]] | None = None
        for detection in ranked_matches:
            pose_estimate = self._estimate_detection_pose(
                detection,
                image,
                preferred_world_xyz=tracked_world_xyz,
            )
            world_xyz = pose_estimate.get("xyz")
            if world_xyz is None:
                continue
            distance = _xy_distance(tracked_world_xyz, world_xyz)
            if best is None or distance < best[0]:
                best = (distance, detection, pose_estimate)

        if best is None:
            return None, {"error": f"no stable '{target_text}' candidate had a 3D pose"}
        if best[0] > _TRACK_REASSOCIATION_DISTANCE_M:
            return (
                None,
                {
                    "error": (
                        f"nearest '{target_text}' candidate moved {best[0]:.2f} m from "
                        "the active track"
                    )
                },
            )
        return best[1], best[2]

    def _update_tracked_detection(self, result: FindObjectDetection) -> FindObjectDetection:
        if result.world_xyz is None:
            return result

        with self._lock:
            previous = self._tracked_world_xyz
            if previous is None:
                tracked = result.world_xyz
            else:
                tracked = tuple(
                    float(prev + _TRACK_SMOOTHING_ALPHA * (new - prev))
                    for prev, new in zip(previous, result.world_xyz, strict=True)
                )
            self._tracked_world_xyz = tracked

        return replace(result, world_xyz=tracked)

    def _prepare_navigation_goal(self, result: FindObjectDetection) -> FindObjectDetection:
        if result.world_xyz is None:
            return replace(
                result,
                navigation_error="object pose unavailable; not sending navigation goal",
            )
        if result.robot_xyz is None:
            return replace(
                result,
                navigation_error="robot odom unavailable; not sending navigation goal",
            )

        nav_goal = _make_standoff_goal(result)
        if nav_goal.get("error"):
            return replace(result, navigation_error=nav_goal["error"])

        return replace(
            result,
            nav_goal_xyz=nav_goal["goal_xyz"],
            nav_goal_yaw=nav_goal["yaw"],
            standoff_m=nav_goal["standoff_m"],
            navigation_error=None,
        )

    def _should_send_goal_update(
        self,
        result: FindObjectDetection,
        *,
        force: bool,
    ) -> bool:
        if result.nav_goal_xyz is None or result.navigation_error:
            return False
        if force:
            return True

        with self._lock:
            last_goal_xyz = self._last_sent_goal_xyz
            last_goal_yaw = self._last_sent_goal_yaw
            last_goal_sent = self._last_goal_sent_monotonic

        if last_goal_xyz is None:
            return True
        if time.monotonic() - last_goal_sent < _GOAL_UPDATE_MIN_INTERVAL_S:
            return False

        goal_delta = _xy_distance(last_goal_xyz, result.nav_goal_xyz)
        yaw_delta = _angle_distance(last_goal_yaw or 0.0, result.nav_goal_yaw or 0.0)
        return goal_delta >= _GOAL_UPDATE_DISTANCE_M or yaw_delta >= _GOAL_UPDATE_YAW_RAD

    def _send_navigation_goal(self, result: FindObjectDetection) -> FindObjectDetection:
        if result.nav_goal_xyz is None:
            return replace(
                result,
                navigation_error="navigation goal unavailable; not sending navigation goal",
            )

        goal = PoseStamped(
            ts=time.time(),
            frame_id=result.world_frame_id or "world",
            position=result.nav_goal_xyz,
            orientation=Quaternion.from_euler(Vector3(0.0, 0.0, result.nav_goal_yaw or 0.0)),
        )

        try:
            set_goal = self.get_rpc_calls("NavigationInterface.set_goal")
            accepted = bool(set_goal(goal))
        except Exception as exc:
            logger.warning("Failed to send object navigation goal", exc_info=True)
            return replace(
                result,
                navigation_error=f"NavigationInterface.set_goal failed: {exc}",
            )

        if accepted:
            with self._lock:
                self._last_sent_goal_xyz = result.nav_goal_xyz
                self._last_sent_goal_yaw = result.nav_goal_yaw
                self._last_goal_sent_monotonic = time.monotonic()
                self._goal_update_count += 1

        return replace(
            result,
            navigation_started=accepted,
            navigation_error=None if accepted else "NavigationInterface.set_goal returned false",
        )

    def _navigation_goal_reached(self) -> bool:
        with self._lock:
            if self._last_sent_goal_xyz is None:
                return False

        try:
            is_goal_reached = self.get_rpc_calls("NavigationInterface.is_goal_reached")
            return bool(is_goal_reached())
        except Exception:
            return False

    def _publish_active_markers(self) -> None:
        with self._lock:
            result = self._last_detection

        if result is None:
            robot_xyz = self._latest_robot_xyz()
            if robot_xyz is None:
                return
            self.object_markers.publish(
                EntityMarkers(
                    markers=[
                        Marker(
                            entity_id="robot",
                            label="",
                            entity_type="person",
                            x=robot_xyz[0],
                            y=robot_xyz[1],
                            z=max(0.05, robot_xyz[2]),
                        )
                    ],
                    ts=time.time(),
                )
            )
            return

        self._publish_object_markers(result)

    def _publish_status(self, event: str, message: str) -> None:
        self.status.publish(
            json.dumps(
                {
                    "ts": time.time(),
                    "event": event,
                    "message": message,
                    "detection": asdict(self._last_detection) if self._last_detection else None,
                    "policy": self._policy_status_snapshot(),
                }
            )
        )

    def _policy_status_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": self._policy_mode,
                "target_text": self._policy_target_text,
                "state_changed_ts": self._policy_state_changed_ts,
                "last_detection_seen_ts": self._last_detection_seen_ts,
                "detection_count": self._detection_count,
                "goal_update_count": self._goal_update_count,
                "tracked_world_xyz": self._tracked_world_xyz,
                "last_sent_goal_xyz": self._last_sent_goal_xyz,
                "last_sent_goal_yaw": self._last_sent_goal_yaw,
                "search_forward_blocked": self._search_forward_blocked,
                "detector_mode": self._detector_mode,
                "manual_motion": self._manual_motion_name,
            }

    def _start_manual_motion(
        self,
        *,
        name: str,
        linear_x: float,
        angular_z: float,
        duration_s: float,
        message: str,
    ) -> str:
        self._stop_manual_motion(join=True)
        self._transition_to_idle(
            f"manual command {name}",
            cancel_navigation=True,
            clear_target=True,
        )

        self._manual_motion_stop_event.clear()
        thread = Thread(
            target=self._manual_motion_loop,
            kwargs={
                "name": name,
                "linear_x": linear_x,
                "angular_z": angular_z,
                "duration_s": duration_s,
            },
            daemon=True,
        )
        with self._lock:
            self._manual_motion_name = name
            self._manual_motion_thread = thread

        thread.start()
        self._publish_status("manual_motion", message)
        return message

    def _manual_motion_loop(
        self,
        *,
        name: str,
        linear_x: float,
        angular_z: float,
        duration_s: float,
    ) -> None:
        twist = Twist(
            Vector3(linear_x, 0.0, 0.0),
            Vector3(0.0, 0.0, angular_z),
        )
        deadline = time.monotonic() + duration_s
        try:
            while (
                time.monotonic() < deadline
                and not self._manual_motion_stop_event.is_set()
                and not self._policy_stop_event.is_set()
            ):
                self.cmd_vel.publish(twist)
                self._manual_motion_stop_event.wait(_MANUAL_TWIST_INTERVAL_S)
        finally:
            self._publish_zero_twist()
            with self._lock:
                if self._manual_motion_name == name:
                    self._manual_motion_name = None
                    self._manual_motion_thread = None

    def _stop_manual_motion(self, *, join: bool) -> None:
        self._manual_motion_stop_event.set()
        with self._lock:
            thread = self._manual_motion_thread
            self._manual_motion_name = None

        if join and thread is not None and thread is not current_thread():
            thread.join(timeout=1.0)

        with self._lock:
            if self._manual_motion_thread is thread:
                self._manual_motion_thread = None

    def _transition_to_idle(
        self,
        reason: str,
        *,
        cancel_navigation: bool,
        clear_target: bool,
    ) -> None:
        with self._lock:
            self._policy_mode = _POLICY_IDLE
            self._policy_state_changed_ts = time.time()
            if clear_target:
                self._policy_target_text = None
                self._tracked_world_xyz = None
                self._last_detection = None
                self._last_detection_seen_ts = None
                self._last_sent_goal_xyz = None
                self._last_sent_goal_yaw = None

        self._publish_zero_twist()
        if clear_target:
            self.object_markers.publish(EntityMarkers(markers=[], ts=time.time()))
        self._publish_mode_only_image()
        if cancel_navigation:
            self._cancel_navigation()
        logger.info("Find-object policy idle", reason=reason)

    def _publish_search_twist(self) -> None:
        forward_blocked = self._is_search_forward_blocked()
        linear_x = _SEARCH_BLOCKED_LINEAR_X_M_S if forward_blocked else _SEARCH_LINEAR_X_M_S
        with self._lock:
            self._search_forward_blocked = forward_blocked

        self.cmd_vel.publish(
            Twist(
                Vector3(linear_x, 0.0, 0.0),
                Vector3(0.0, 0.0, _SEARCH_YAW_RATE_RAD_S),
            )
        )

    def _publish_zero_twist(self) -> None:
        self.cmd_vel.publish(Twist.zero())

    def _cancel_navigation(self) -> None:
        try:
            cancel_goal = self.get_rpc_calls("NavigationInterface.cancel_goal")
        except Exception:
            logger.warning("Navigation cancel is not connected")
            return
        cancel_goal()

    def _is_search_forward_blocked(self) -> bool:
        with self._lock:
            costmap = self._latest_global_costmap
            odom = self._latest_odom

        if costmap is None or odom is None:
            return False

        yaw = odom.orientation.euler[2]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        robot_x = odom.position.x
        robot_y = odom.position.y

        # Small fan in front of the body. Unknown cells are ignored so initial
        # exploration does not freeze before the local map fills in.
        for forward_m in (0.35, 0.55, 0.8):
            for lateral_m in (-0.18, 0.0, 0.18):
                x = robot_x + cos_yaw * forward_m - sin_yaw * lateral_m
                y = robot_y + sin_yaw * forward_m + cos_yaw * lateral_m
                value = costmap.cell_value(Vector3(x, y, 0.0))
                if value != CostValues.UNKNOWN and value >= _SEARCH_BLOCKED_OCCUPANCY_THRESHOLD:
                    return True
        return False

    def _latest_robot_xyz(self) -> tuple[float, float, float] | None:
        with self._lock:
            odom = self._latest_odom
        if odom is None:
            return None
        return (
            float(odom.position.x),
            float(odom.position.y),
            float(odom.position.z),
        )

    def _execute_find_object(self, target_text: str) -> str:
        image = self._latest_image
        if image is None:
            self._last_detection = None
            return (
                f"Accepted object navigation target: '{target_text}', but no camera image has "
                "arrived yet. Wait a few seconds after starting the sim and try again."
            )

        detector = self._get_detector()
        image_detections = detector.process_image(image)
        detections = list(image_detections)
        detected_labels = sorted({detection.name for detection in detections})
        matches = [
            detection
            for detection in detections
            if _target_matches_detection(target_text, detection.name)
        ]

        if not matches:
            self._last_detection = None
            self._publish_unmatched_image(image, target_text, detected_labels)
            labels = ", ".join(detected_labels) if detected_labels else "none"
            return (
                f"No '{target_text}' detection in the latest {image.width}x{image.height} image. "
                f"YOLO labels currently visible: {labels}."
            )

        chosen = max(matches, key=_detection_area_px)
        pose_estimate = self._estimate_detection_pose(chosen, image)
        result = _detection_to_result(
            target_text=target_text,
            detection=chosen,
            image=image,
            matching_count=len(matches),
            detected_labels=detected_labels,
            pose_estimate=pose_estimate,
            robot_pose=self._latest_odom,
        )
        result = self._navigate_to_detection(result)
        self._last_detection = result
        self._publish_object_markers(result)
        self._publish_annotated_image(image, chosen, result)

        pose_text = (
            f" World xyz=({result.world_xyz[0]:.2f}, {result.world_xyz[1]:.2f}, {result.world_xyz[2]:.2f})"
            if result.world_xyz
            else f" Pose unavailable: {result.pose_method}."
        )
        nav_text = _navigation_message(result)
        pose_source = "segmentation mask" if result.segmentation_used else "bbox"
        return (
            f"Detected '{result.detected_label}' for target '{target_text}' "
            f"at bbox {tuple(round(value, 1) for value in result.bbox_xyxy)} "
            f"(confidence {result.confidence:.2f}, area {result.area_px:.0f}px). "
            f"Picked the largest of {result.matching_count} matching detection(s); "
            f"pose source: {pose_source}. "
            f"{pose_text} {nav_text}"
        )

    def _execute_stop(self) -> str:
        self._stop_manual_motion(join=True)
        self._transition_to_idle("stop command", cancel_navigation=True, clear_target=True)
        self._publish_status("stopped", "Stop command accepted.")
        return "Stop command accepted; object-search policy and navigation were cancelled."

    def _get_detector(self) -> Any:
        if self._detector is None:
            try:
                from dimos.perception.detection.detectors.yolo_seg import YoloSeg2DDetector

                self._detector = YoloSeg2DDetector()
                self._detector_mode = "segmentation"
            except Exception:
                logger.warning(
                    "YOLO segmentation detector unavailable; falling back to bbox detector",
                    exc_info=True,
                )
                from dimos.perception.detection.detectors.yolo import Yolo2DDetector

                self._detector = Yolo2DDetector()
                self._detector_mode = "bbox"
        return self._detector

    def _estimate_detection_pose(
        self,
        detection: Any,
        image: Image,
        *,
        preferred_world_xyz: tuple[float, float, float] | None = None,
    ) -> dict[str, Any]:
        camera_info = self._latest_camera_info
        pointcloud = self._latest_global_map

        if camera_info is None:
            return {"error": "camera_info has not arrived yet"}
        if pointcloud is None or len(pointcloud) == 0:
            return {"error": "global_map pointcloud has not arrived yet"}

        try:
            transform = self.tf.get("camera_optical", pointcloud.frame_id, image.ts, 1.0)
        except Exception as exc:
            return {"error": f"camera/world transform lookup failed: {exc}"}

        if not transform:
            return {
                "error": (
                    f"no transform from {pointcloud.frame_id!r} to camera_optical "
                    f"near image ts={image.ts:.3f}"
                )
            }

        try:
            from dimos.perception.detection.type.detection3d.pointcloud import Detection3DPC

            uses_mask = _has_segmentation_mask(detection)
            mask_part = "segmentation_mask" if uses_mask else "bbox"
            detection3d = Detection3DPC.from_2d(
                detection,
                world_pointcloud=pointcloud,
                camera_info=camera_info,
                world_to_optical_transform=transform,
            )
            method = f"projected_global_map_{mask_part}_filtered"
            if detection3d is None:
                detection3d = Detection3DPC.from_2d(
                    detection,
                    world_pointcloud=pointcloud,
                    camera_info=camera_info,
                    world_to_optical_transform=transform,
                    filters=[],
                )
                method = f"projected_global_map_{mask_part}_unfiltered"
            if detection3d is None:
                return {"error": "no map points projected inside bbox"}

            points, _ = detection3d.pointcloud.as_numpy()
            cluster = _select_pose_cluster(points, preferred_world_xyz)
            if cluster is not None:
                center_xyz = cluster["center_xyz"]
                selected_point_count = cluster["selected_point_count"]
                cluster_count = cluster["cluster_count"]
                if cluster_count > 1:
                    method = f"{method}_clustered"
            else:
                center = detection3d.pointcloud.center
                center_xyz = (center.x, center.y, center.z)
                selected_point_count = len(detection3d.pointcloud)
                cluster_count = 1

            return {
                "xyz": center_xyz,
                "frame_id": detection3d.pointcloud.frame_id,
                "method": method,
                "point_count": len(detection3d.pointcloud),
                "selected_point_count": selected_point_count,
                "cluster_count": cluster_count,
                "ambiguity_handled": cluster_count > 1,
            }
        except Exception as exc:
            logger.warning("Failed to estimate 3D object pose", exc_info=True)
            return {"error": f"3D projection failed: {exc}"}

    def _publish_annotated_image(
        self,
        image: Image,
        detection: Any,
        result: FindObjectDetection,
    ) -> None:
        try:
            frame = image.to_bgr().to_opencv().copy()
            _draw_segmentation_overlay(frame, detection)
            _draw_mode_banner(frame, self._policy_visual_label())
            self.detected_image.publish(
                Image.from_opencv(
                    frame,
                    format=ImageFormat.BGR,
                    frame_id=image.frame_id,
                    ts=image.ts,
                )
            )
        except Exception:
            logger.warning("Failed to publish annotated find-object image", exc_info=True)

    def _publish_object_markers(self, result: FindObjectDetection) -> None:
        markers: list[Marker] = []
        robot_xyz = self._latest_robot_xyz() or result.robot_xyz
        if robot_xyz:
            rx, ry, rz = robot_xyz
            markers.append(
                Marker(
                    entity_id="robot",
                    label="",
                    entity_type="person",
                    x=rx,
                    y=ry,
                    z=max(0.05, rz),
                )
            )
        if result.world_xyz:
            x, y, z = result.world_xyz
            markers.append(
                Marker(
                    entity_id="target",
                    label="",
                    entity_type="location",
                    x=x,
                    y=y,
                    z=z,
                )
            )
        if result.nav_goal_xyz:
            x, y, z = result.nav_goal_xyz
            markers.append(
                Marker(
                    entity_id="nav_goal",
                    label="",
                    entity_type="object",
                    x=x,
                    y=y,
                    z=max(0.05, z),
                )
            )
        if markers:
            self.object_markers.publish(EntityMarkers(markers=markers, ts=time.time()))

    def _navigate_to_detection(self, result: FindObjectDetection) -> FindObjectDetection:
        planned = self._prepare_navigation_goal(result)
        if planned.navigation_error:
            return planned
        return self._send_navigation_goal(planned)

    def _publish_unmatched_image(
        self,
        image: Image,
        target_text: str,
        _detected_labels: list[str],
    ) -> None:
        try:
            frame = image.to_bgr().to_opencv().copy()
            _draw_mode_banner(frame, self._policy_visual_label())
            self.detected_image.publish(
                Image.from_opencv(
                    frame,
                    format=ImageFormat.BGR,
                    frame_id=image.frame_id,
                    ts=image.ts,
                )
            )
        except Exception:
            logger.warning("Failed to publish unmatched find-object image", exc_info=True)

    def _publish_mode_only_image(self) -> None:
        with self._lock:
            image = self._latest_image

        if image is None:
            return

        try:
            frame = image.to_bgr().to_opencv().copy()
            _draw_mode_banner(frame, self._policy_visual_label())
            self.detected_image.publish(
                Image.from_opencv(
                    frame,
                    format=ImageFormat.BGR,
                    frame_id=image.frame_id,
                    ts=image.ts,
                )
            )
        except Exception:
            logger.warning("Failed to publish find-object mode image", exc_info=True)

    def _policy_visual_label(self) -> str:
        with self._lock:
            mode = self._policy_mode
            target_text = self._policy_target_text

        if target_text:
            return f"{mode.upper()} | {target_text}"
        return mode.upper()


def parse_find_object_command(text: str) -> FindObjectPlan:
    raw_text = text or ""
    normalized = _normalize(raw_text)

    if not normalized:
        return FindObjectPlan(
            intent="unknown",
            raw_text=raw_text,
            accepted=False,
            reason="empty command",
        )

    if normalized in _STOP_COMMANDS:
        return FindObjectPlan(intent="stop", raw_text=raw_text, accepted=True)

    target = _extract_target(normalized)
    if target:
        return FindObjectPlan(
            intent="find_object",
            raw_text=raw_text,
            target_text=target,
            accepted=True,
        )

    return FindObjectPlan(
        intent="unknown",
        raw_text=raw_text,
        accepted=False,
        reason="no object target found",
    )


def _normalize(text: str) -> str:
    compact = re.sub(r"\s+", " ", text.strip().lower())
    return compact.strip(" .!?")


def _extract_target(text: str) -> str | None:
    for prefix in _FIND_PREFIXES:
        if text == prefix:
            return None
        if text.startswith(prefix + " "):
            return _clean_target(text[len(prefix) :])

    # Bare noun phrases are useful during development: "red soda can".
    if 1 <= len(text.split()) <= 10:
        return _clean_target(text)

    return None


def _clean_target(text: str) -> str:
    target = text.strip()
    for article in ("the ", "a ", "an "):
        if target.startswith(article):
            target = target[len(article) :]
            break
    return target.strip()


def _target_matches_detection(target_text: str, detected_label: str) -> bool:
    target_tokens = _label_tokens(target_text)
    label_tokens = _label_tokens(detected_label)
    if not target_tokens or not label_tokens:
        return False

    # For phrases like "red chair", the detector's "chair" class should match.
    return bool(target_tokens & label_tokens) or detected_label.lower() in target_text.lower()


def _label_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")))
    singulars = {token[:-1] for token in tokens if token.endswith("s") and len(token) > 3}
    return tokens | singulars


def _has_segmentation_mask(detection: Any) -> bool:
    mask = getattr(detection, "mask", None)
    if mask is None:
        return False
    try:
        return bool((mask > 0).any())
    except Exception:
        return False


def _detection_area_px(detection: Any) -> float:
    mask = getattr(detection, "mask", None)
    if mask is not None:
        try:
            return float((mask > 0).sum())
        except Exception:
            pass
    return float(detection.bbox_2d_volume())


def _xy_distance(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _angle_distance(a: float, b: float) -> float:
    delta = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(delta)


def _select_pose_cluster(
    points: Any,
    preferred_world_xyz: tuple[float, float, float] | None,
) -> dict[str, Any] | None:
    import numpy as np

    points_np = np.asarray(points, dtype=float)
    if points_np.ndim != 2 or points_np.shape[1] < 3 or len(points_np) == 0:
        return None

    if len(points_np) < _POSE_CLUSTER_MIN_POINTS:
        center = tuple(float(value) for value in points_np[:, :3].mean(axis=0))
        return {
            "center_xyz": center,
            "selected_point_count": len(points_np),
            "cluster_count": 1,
        }

    cells = np.floor(points_np[:, :2] / _POSE_CLUSTER_CELL_M).astype(int)
    cell_to_indices: dict[tuple[int, int], list[int]] = {}
    for idx, cell in enumerate(cells):
        key = (int(cell[0]), int(cell[1]))
        cell_to_indices.setdefault(key, []).append(idx)

    visited: set[tuple[int, int]] = set()
    clusters: list[np.ndarray[Any, np.dtype[np.int64]]] = []
    for start_cell in cell_to_indices:
        if start_cell in visited:
            continue

        stack = [start_cell]
        visited.add(start_cell)
        cluster_indices: list[int] = []

        while stack:
            cell = stack.pop()
            cluster_indices.extend(cell_to_indices[cell])
            cx, cy = cell
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor = (cx + dx, cy + dy)
                    if neighbor in visited or neighbor not in cell_to_indices:
                        continue
                    visited.add(neighbor)
                    stack.append(neighbor)

        clusters.append(np.array(cluster_indices, dtype=np.int64))

    if not clusters:
        return None

    clusters.sort(key=len, reverse=True)
    min_points = max(
        _POSE_CLUSTER_MIN_POINTS,
        int(len(clusters[0]) * _POSE_CLUSTER_MIN_RELATIVE_TO_LARGEST),
    )
    significant_clusters = [cluster for cluster in clusters if len(cluster) >= min_points]
    if not significant_clusters:
        significant_clusters = [clusters[0]]

    centers = [points_np[cluster, :3].mean(axis=0) for cluster in significant_clusters]
    if preferred_world_xyz is not None:
        preferred_xy = np.array(preferred_world_xyz[:2], dtype=float)
        selected_idx = min(
            range(len(significant_clusters)),
            key=lambda idx: float(np.linalg.norm(centers[idx][:2] - preferred_xy)),
        )
    else:
        selected_idx = 0

    selected_cluster = significant_clusters[selected_idx]
    center = tuple(float(value) for value in centers[selected_idx])
    return {
        "center_xyz": center,
        "selected_point_count": len(selected_cluster),
        "cluster_count": len(significant_clusters),
    }


def _detection_to_result(
    *,
    target_text: str,
    detection: Any,
    image: Image,
    matching_count: int,
    detected_labels: list[str],
    pose_estimate: dict[str, Any],
    robot_pose: PoseStamped | None,
) -> FindObjectDetection:
    x1, y1, x2, y2 = (float(value) for value in detection.bbox)
    center_x, center_y = detection.center_bbox
    world_xyz = pose_estimate.get("xyz")
    if world_xyz is not None:
        world_xyz = tuple(float(value) for value in world_xyz)
    robot_xyz = None
    if robot_pose is not None:
        robot_xyz = (
            float(robot_pose.position.x),
            float(robot_pose.position.y),
            float(robot_pose.position.z),
        )
    return FindObjectDetection(
        target_text=target_text,
        detected_label=str(detection.name),
        confidence=float(detection.confidence),
        bbox_xyxy=(x1, y1, x2, y2),
        center_xy=(float(center_x), float(center_y)),
        area_px=float(_detection_area_px(detection)),
        image_width=image.width,
        image_height=image.height,
        matching_count=matching_count,
        detected_labels=detected_labels,
        world_xyz=world_xyz,
        world_frame_id=pose_estimate.get("frame_id"),
        pose_method=pose_estimate.get("method") or pose_estimate.get("error"),
        object_point_count=pose_estimate.get("selected_point_count")
        or pose_estimate.get("point_count"),
        pose_cluster_count=pose_estimate.get("cluster_count"),
        pose_ambiguity_handled=bool(pose_estimate.get("ambiguity_handled")),
        robot_xyz=robot_xyz,
        segmentation_used=_has_segmentation_mask(detection),
    )


def _draw_outlined_text(
    frame: Any,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    *,
    scale: float,
) -> None:
    import cv2

    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_mode_banner(frame: Any, label: str) -> None:
    import cv2

    label = _truncate_visual_label(label)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.68
    thickness = 2
    padding_x = 12
    padding_y = 9
    text_size, baseline = cv2.getTextSize(label, font, scale, thickness)
    width = min(frame.shape[1], text_size[0] + padding_x * 2)
    height = text_size[1] + padding_y * 2 + baseline

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, dst=frame)
    cv2.putText(
        frame,
        label,
        (padding_x, padding_y + text_size[1]),
        font,
        scale,
        (245, 245, 245),
        thickness,
        cv2.LINE_AA,
    )


def _truncate_visual_label(label: str, max_len: int = 46) -> str:
    if len(label) <= max_len:
        return label
    return label[: max_len - 3].rstrip() + "..."


def _draw_segmentation_overlay(frame: Any, detection: Any) -> None:
    mask = getattr(detection, "mask", None)
    if mask is None:
        return

    try:
        import cv2
        import numpy as np

        mask_np = np.asarray(mask)
        if mask_np.shape[:2] != frame.shape[:2]:
            mask_np = cv2.resize(
                mask_np.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        active = mask_np > 0
        if not active.any():
            return

        overlay = frame.copy()
        overlay[active] = (30, 210, 255)
        frame[active] = cv2.addWeighted(frame[active], 0.55, overlay[active], 0.45, 0)

        contours, _ = cv2.findContours(
            mask_np.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(frame, contours, -1, (30, 210, 255), 2, cv2.LINE_AA)
    except Exception:
        logger.warning("Failed to draw segmentation overlay", exc_info=True)


def _make_standoff_goal(
    result: FindObjectDetection,
    *,
    desired_standoff_m: float = 1.6,
    minimum_standoff_m: float = 0.9,
) -> dict[str, Any]:
    if result.world_xyz is None or result.robot_xyz is None:
        return {"error": "missing object or robot position"}

    target_x, target_y, _target_z = result.world_xyz
    robot_x, robot_y, robot_z = result.robot_xyz
    dx = target_x - robot_x
    dy = target_y - robot_y
    distance = math.hypot(dx, dy)
    if distance < 1e-3:
        return {"error": "robot and object positions are too close to compute stand-off"}

    standoff = min(desired_standoff_m, max(minimum_standoff_m, distance * 0.65))
    if distance <= standoff + 0.05:
        goal_x = robot_x
        goal_y = robot_y
        standoff = distance
    else:
        unit_x = dx / distance
        unit_y = dy / distance
        goal_x = target_x - unit_x * standoff
        goal_y = target_y - unit_y * standoff

    yaw_to_target = math.atan2(target_y - goal_y, target_x - goal_x)
    return {
        "goal_xyz": (float(goal_x), float(goal_y), float(robot_z)),
        "yaw": float(yaw_to_target),
        "standoff_m": float(standoff),
    }


def _navigation_message(result: FindObjectDetection) -> str:
    if result.navigation_started and result.nav_goal_xyz:
        gx, gy, gz = result.nav_goal_xyz
        yaw = result.nav_goal_yaw if result.nav_goal_yaw is not None else 0.0
        standoff = result.standoff_m if result.standoff_m is not None else 0.0
        return (
            f"Navigation goal sent: ({gx:.2f}, {gy:.2f}, {gz:.2f}), "
            f"yaw {yaw:.2f} rad, stand-off {standoff:.2f} m. "
            "Planner should rotate to face the object at arrival."
        )
    if result.navigation_error:
        return f"Navigation not started: {result.navigation_error}."
    return "Navigation not started."


find_object_task = FindObjectTask.blueprint

__all__ = [
    "FindObjectDetection",
    "FindObjectTask",
    "find_object_task",
    "parse_find_object_command",
]
