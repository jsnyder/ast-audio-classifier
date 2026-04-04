"""Cross-camera event consolidation for notification dedup.

Groups detections of the same event label group across cameras within a
sliding time window.  Per-camera binary sensors continue firing independently;
the consolidator produces *additional* consolidated sensors for automations
that want a single trigger per physical event.

Safety-critical groups (smoke_alarm, glass_break, siren, screaming, crying)
fire immediately with no dedup delay.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Groups that bypass the dedup window and fire immediately
SAFETY_GROUPS = frozenset({
    "smoke_alarm",
    "glass_break",
    "siren",
    "screaming",
    "crying",
})


@dataclass
class ConsolidatedEpisode:
    """A single consolidated event episode spanning one or more cameras."""

    group: str
    cameras: list[str] = field(default_factory=list)
    max_confidence: float = 0.0
    detection_count: int = 0
    first_detected: float = 0.0  # monotonic trigger_time
    last_detected: float = 0.0   # monotonic trigger_time
    published: bool = False


class EventConsolidator:
    """Aggregates detections across cameras into consolidated events.

    Args:
        window_seconds: Time window for grouping detections of the same group.
        auto_off_seconds: Base auto-off delay for consolidated sensors.
        on_consolidated: Callback invoked when a consolidated event should be
            published. Signature: (group, episode) -> None.
    """

    def __init__(
        self,
        window_seconds: float = 5.0,
        auto_off_seconds: int = 30,
        on_consolidated: Callable[[str, ConsolidatedEpisode], None] | None = None,
    ) -> None:
        self._window = window_seconds
        self._auto_off = auto_off_seconds
        self._on_consolidated = on_consolidated
        # group -> list of active episodes
        self._episodes: dict[str, list[ConsolidatedEpisode]] = {}

    @property
    def auto_off_seconds(self) -> int:
        """Consolidated auto-off = base + 10 to outlast per-camera sensors."""
        return self._auto_off + 10

    def report_detection(
        self,
        camera_name: str,
        group: str,
        confidence: float,
        trigger_time: float,
    ) -> None:
        """Report a detection from a camera.

        Uses trigger_time (monotonic) rather than wall-clock time to correctly
        group events even when AST inference is serialized across cameras.
        """
        # Safety-critical groups bypass dedup — always fire immediately
        if group in SAFETY_GROUPS:
            ep = ConsolidatedEpisode(
                group=group,
                cameras=[camera_name],
                max_confidence=confidence,
                detection_count=1,
                first_detected=trigger_time,
                last_detected=trigger_time,
            )
            self._episodes.setdefault(group, []).append(ep)
            self._publish(group, ep)
            return

        episodes = self._episodes.setdefault(group, [])

        # Find an existing episode within the window (abs handles out-of-order arrivals)
        matched = None
        for ep in episodes:
            if abs(trigger_time - ep.last_detected) <= self._window:
                matched = ep
                break

        if matched is not None:
            # Merge into existing episode
            camera_is_new = camera_name not in matched.cameras
            if camera_is_new:
                matched.cameras.append(camera_name)
            matched.max_confidence = max(matched.max_confidence, confidence)
            matched.detection_count += 1
            matched.last_detected = max(matched.last_detected, trigger_time)

            # Re-publish only when a new camera joins the episode
            if not matched.published or camera_is_new:
                self._publish(group, matched)
        else:
            # New episode
            ep = ConsolidatedEpisode(
                group=group,
                cameras=[camera_name],
                max_confidence=confidence,
                detection_count=1,
                first_detected=trigger_time,
                last_detected=trigger_time,
            )
            episodes.append(ep)
            self._publish(group, ep)

    def _publish(self, group: str, episode: ConsolidatedEpisode) -> None:
        """Invoke the consolidated callback."""
        episode.published = True
        if self._on_consolidated is not None:
            self._on_consolidated(group, episode)

    def cleanup_stale(self) -> None:
        """Remove episodes older than auto_off_seconds + window.

        Call periodically (e.g. every 10s) to prevent memory growth.
        """
        now = time.monotonic()
        cutoff = self._auto_off + self._window + 10
        for group in list(self._episodes):
            episodes = self._episodes[group]
            self._episodes[group] = [
                ep for ep in episodes if now - ep.last_detected < cutoff
            ]
            if not self._episodes[group]:
                del self._episodes[group]
