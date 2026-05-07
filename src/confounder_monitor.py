"""Monitor HA entity state to track active confounders per camera.

Polls the HA Supervisor REST API for entity states and evaluates
confounder conditions (e.g. "!off", ">200", "playing").

When a confounder is active, its confused_groups are flagged — not
suppressed — so downstream consumers can decide how to handle them.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import CameraConfig, ConfounderConfig

logger = logging.getLogger(__name__)

# Regex patterns for active_when expressions
_NEGATE_RE = re.compile(r"^!(.+)$")  # "!off" → state != "off"
_GT_RE = re.compile(r"^>([\d.]+)$")  # ">200" → float(state) > 200
_GTE_RE = re.compile(r"^>=([\d.]+)$")  # ">=200" → float(state) >= 200
_LT_RE = re.compile(r"^<([\d.]+)$")  # "<50" → float(state) < 50
_LTE_RE = re.compile(r"^<=([\d.]+)$")  # "<=50" → float(state) <= 50


def evaluate_condition(active_when: str, state: str) -> bool:
    """Evaluate an active_when condition against an entity state string.

    Supported expressions:
        "!off"    → True when state != "off"
        ">200"    → True when float(state) > 200
        ">=200"   → True when float(state) >= 200
        "<50"     → True when float(state) < 50
        "<=50"    → True when float(state) <= 50
        "playing" → True when state == "playing" (exact match)
    """
    if state in ("unavailable", "unknown"):
        return False

    m = _NEGATE_RE.match(active_when)
    if m:
        return state != m.group(1)

    # Order matters: >= before >, <= before < (longer prefix must match first)
    m = _GTE_RE.match(active_when)
    if m:
        try:
            return float(state) >= float(m.group(1))
        except ValueError:
            return False

    m = _GT_RE.match(active_when)
    if m:
        try:
            return float(state) > float(m.group(1))
        except ValueError:
            return False

    m = _LTE_RE.match(active_when)
    if m:
        try:
            return float(state) <= float(m.group(1))
        except ValueError:
            return False

    m = _LT_RE.match(active_when)
    if m:
        try:
            return float(state) < float(m.group(1))
        except ValueError:
            return False

    # Exact match (e.g. "playing", "on")
    return state == active_when


class ConfounderMonitor:
    """Polls HA entity states and tracks which confounders are active per camera."""

    def __init__(
        self,
        cameras: list[CameraConfig],
        poll_interval: float = 10.0,
    ) -> None:
        self._poll_interval = poll_interval
        self._task: asyncio.Task | None = None

        # Build lookup: camera_name -> list of confounders
        self._camera_confounders: dict[str, list[ConfounderConfig]] = {}
        # Collect unique entity IDs to poll
        self._entity_ids: set[str] = set()

        for cam in cameras:
            if cam.confounders:
                self._camera_confounders[cam.name] = list(cam.confounders)
                for c in cam.confounders:
                    self._entity_ids.add(c.entity_id)

        # Current state cache: entity_id -> state string
        self._entity_states: dict[str, str] = {}

        # Supervisor API access
        self._supervisor_token = os.environ.get("SUPERVISOR_TOKEN")
        self._available = bool(self._supervisor_token)

        if not self._available:
            logger.warning(
                "SUPERVISOR_TOKEN not set — confounder monitoring disabled (standalone Docker mode)"
            )

    @property
    def available(self) -> bool:
        """Whether the monitor can poll entity states."""
        return self._available

    async def start(self) -> None:
        """Start the background polling task (idempotent)."""
        if not self._available or not self._entity_ids:
            return
        if self._task and not self._task.done():
            return
        # Do one immediate poll before starting the loop
        await self._poll_states()
        self._task = asyncio.create_task(self._poll_loop(), name="confounder-monitor")
        logger.info(
            "Confounder monitor started: tracking %d entities for %d cameras",
            len(self._entity_ids),
            len(self._camera_confounders),
        )

    async def stop(self) -> None:
        """Stop the background polling task."""
        task = self._task
        self._task = None
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _poll_loop(self) -> None:
        """Periodically poll entity states on a monotonic deadline."""
        import time

        interval = self._poll_interval
        next_deadline = time.monotonic() + interval
        while True:
            now = time.monotonic()
            sleep_for = max(0.0, next_deadline - now)
            await asyncio.sleep(sleep_for)
            next_deadline += interval
            try:
                await self._poll_states()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error polling confounder entity states")

    async def _poll_states(self) -> None:
        """Fetch current state for all monitored entities from HA API."""
        await asyncio.to_thread(self._poll_states_sync)

    def _fetch_entity_state(self, entity_id: str) -> tuple[str, str]:
        """Fetch a single entity state. Returns (entity_id, state)."""
        import json
        import urllib.request

        headers = {"Authorization": f"Bearer {self._supervisor_token}"}
        url = f"http://supervisor/core/api/states/{entity_id}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            return entity_id, data.get("state", "unavailable")
        except Exception:
            logger.debug("Failed to fetch %s", entity_id, exc_info=True)
            return entity_id, "unavailable"

    def _poll_states_sync(self) -> None:
        """Synchronous polling — runs in a thread to avoid blocking the event loop.

        Fetches confounder entities concurrently and atomically replaces
        the state dict so readers on the event loop always see a consistent snapshot.
        """
        from concurrent.futures import ThreadPoolExecutor

        try:
            entities = list(self._entity_ids)
            if not entities:
                return
            with ThreadPoolExecutor(max_workers=min(len(entities), 8)) as pool:
                results = pool.map(self._fetch_entity_state, entities)

            new_states: dict[str, str] = {}
            for entity_id, new_state in results:
                old_state = self._entity_states.get(entity_id)
                if old_state != new_state:
                    logger.debug(
                        "Confounder entity %s: %s -> %s",
                        entity_id,
                        old_state,
                        new_state,
                    )
                new_states[entity_id] = new_state
            # Atomic replacement — readers always see a consistent snapshot
            self._entity_states = new_states
        except Exception:
            logger.debug("Failed to poll entity states from Supervisor API", exc_info=True)

    def get_active_confounders(self, camera_name: str) -> list[ConfounderConfig]:
        """Return confounders whose condition is currently met for this camera."""
        confounders = self._camera_confounders.get(camera_name, [])
        active = []
        for c in confounders:
            state = self._entity_states.get(c.entity_id, "unavailable")
            if evaluate_condition(c.active_when, state):
                active.append(c)
        return active

    def get_confused_groups(self, camera_name: str) -> frozenset[str]:
        """Return the union of confused_groups from all active confounders for this camera."""
        active = self.get_active_confounders(camera_name)
        if not active:
            return frozenset()
        groups: set[str] = set()
        for c in active:
            groups.update(c.confused_groups)
        return frozenset(groups)

    def get_confounder_context(self, camera_name: str, group: str) -> dict | None:
        """Return confounder context if a group is confounded, else None.

        Returns: {"entity_id": str, "state": str, "active_when": str}
        """
        confounders = self._camera_confounders.get(camera_name, [])
        for c in confounders:
            if group not in c.confused_groups:
                continue
            state = self._entity_states.get(c.entity_id, "unavailable")
            if evaluate_condition(c.active_when, state):
                return {
                    "entity_id": c.entity_id,
                    "state": state,
                    "active_when": c.active_when,
                }
        return None
