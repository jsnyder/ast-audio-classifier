"""Noise stress score for audio sensory sensitivity monitoring.

Computes a 0-100 noise stress score from ambient dB levels and classified
audio events, calibrated for ADHD/autism sensory profiles (sensitization,
focus-disruption, overload cascade, extended recovery).

Three components:
  - Ambient (20%): normalized max-across-cameras EMA dB
  - Events (45%): exponentially-decaying impulse accumulator per detection
  - Sustained (35%): slow-attack/very-slow-release EMA (sensory fatigue proxy)

Overload cascade: when sustained > 50, all sounds hit harder (up to 2x).
Masking/calming sounds lose effectiveness during extreme overload (> 85).

Daily tracking records min/max/avg per day for historical trend analysis.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stress tier weights — ADHD/autism sensory profile calibration
# ---------------------------------------------------------------------------
# Based on Gemini 3.1 Pro consultation re: ADHD/autism sensory processing:
#   - Sensitization > habituation: repeated exposure INCREASES distress
#   - Focus disruption: unpredictable interrupts break hyperfocus states
#   - Overload cascade: once overloaded, ALL sounds hit harder
#   - Extended recovery: sensory fatigue takes much longer to dissipate
#
# HIGH STARTLE / SENSITIZATION (3.0): sudden, triggers sensitization loop
# FOCUS DISRUPTION (2.5): unpredictable, breaks concentration/hyperfocus
# LOUD BUT PREDICTABLE (1.2): high volume but pattern-recognizable
# BACKGROUND / MILD (0.4): generally predictable, low attention-capture
# STIMMING / MASKING (-0.5): calming/regulating sounds (conditional)
# DEFAULT: minimal stress contribution

STRESS_TIERS: dict[str, float] = {
    # HIGH STARTLE / SENSITIZATION (3.0)
    "dog_bark": 3.0,
    "screaming": 3.0,
    "crying": 3.0,
    "siren": 3.0,
    "alarm_beep": 3.0,
    "glass_break": 3.0,
    "gunshot_explosion": 3.0,
    # FOCUS DISRUPTION / UNPREDICTABLE (2.5)
    "speech": 2.5,
    "knock": 2.5,
    "doorbell": 2.5,
    # LOUD BUT PREDICTABLE (1.2)
    "vacuum_cleaner": 1.2,
    "power_tools": 1.2,
    "car_horn": 1.2,
    "music": 1.2,
    # BACKGROUND / MILD (0.4)
    "footsteps": 0.4,
    "door": 0.4,
    "cabinet": 0.4,
    "kitchen_appliance": 0.4,
    "vehicle": 0.4,
    "aircraft": 0.4,
    # STIMMING / MASKING (-0.5) — conditional, disabled during extreme overload
    "rain_storm": -0.5,
    "hvac_mechanical": -0.5,
    "water_running": -0.5,
}

# Groups considered HIGH tier for calming suppression
_HIGH_TIER_GROUPS = frozenset(
    group for group, weight in STRESS_TIERS.items() if weight >= 3.0
)

DEFAULT_TIER_WEIGHT = 0.3

# Overload cascade thresholds
OVERLOAD_ONSET = 50.0  # sustained EMA above this triggers overload multiplier
OVERLOAD_RANGE = 50.0  # maps [OVERLOAD_ONSET, OVERLOAD_ONSET+OVERLOAD_RANGE] to [1.0, 2.0]
EXTREME_OVERLOAD = 85.0  # above this, masking/calming sounds stop working


def get_tier_weight(group: str) -> float:
    """Return the stress tier weight for a label group."""
    return STRESS_TIERS.get(group, DEFAULT_TIER_WEIGHT)


# ---------------------------------------------------------------------------
# Ambient normalization constants
# ---------------------------------------------------------------------------
QUIET_FLOOR_DB = -55.0
LOUD_CEILING_DB = -15.0
INDOOR_MULTIPLIER = 1.3

# ---------------------------------------------------------------------------
# Event decay constants — extended for ADHD/autism recovery profile
# ---------------------------------------------------------------------------
DEFAULT_HALF_LIFE = 180.0  # seconds (was 120; longer decay = events linger)
DEFAULT_SATURATION = 25.0  # (was 10; wider dynamic range before ceiling)

# Loudness factor: map trigger dB to [0.3, 2.0]
LOUDNESS_DB_MIN = -50.0
LOUDNESS_DB_MAX = -10.0
LOUDNESS_FACTOR_MIN = 0.3
LOUDNESS_FACTOR_MAX = 2.0

# Multi-camera boost
CAMERA_FACTOR_BASE = 1.0
CAMERA_FACTOR_PER_EXTRA = 0.15

# Event buffer pruning: discard events older than this
EVENT_MAX_AGE = 600.0  # 10 minutes


@dataclass
class StressEvent:
    """A single recorded detection for stress scoring."""

    timestamp: float  # monotonic time
    group: str
    trigger_db: float
    camera: str
    confidence: float  # used as impulse multiplier
    num_cameras: int = 1


@dataclass
class DailyStats:
    """Accumulated statistics for a single day."""

    date: date
    min_score: float = 100.0
    max_score: float = 0.0
    sum_score: float = 0.0
    count: int = 0
    peak_stressor: str | None = None
    peak_stressor_score: float = 0.0

    def record(self, score: float, top_stressor: str | None, top_stressor_weight: float = 0.0) -> None:
        """Record a score sample for this day."""
        self.min_score = min(self.min_score, score)
        self.max_score = max(self.max_score, score)
        self.sum_score += score
        self.count += 1
        if top_stressor and top_stressor_weight > self.peak_stressor_score:
            self.peak_stressor = top_stressor
            self.peak_stressor_score = top_stressor_weight

    @property
    def avg_score(self) -> float:
        return round(self.sum_score / self.count, 1) if self.count > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "min": round(self.min_score, 1) if self.count > 0 else 0.0,
            "max": round(self.max_score, 1),
            "avg": self.avg_score,
            "samples": self.count,
            "peak_stressor": self.peak_stressor,
        }


# Maximum number of daily history entries to keep
DAILY_HISTORY_MAX = 7


class NoiseStressScorer:
    """Computes a noise stress score (0-100) from ambient and event data.

    Args:
        half_life: Exponential decay half-life in seconds.
        saturation: Saturation constant for the event component.
        indoor_cameras: Set of camera names considered indoor (get multiplier).
        update_interval: Seconds between compute() calls.
    """

    def __init__(
        self,
        half_life: float = DEFAULT_HALF_LIFE,
        saturation: float = DEFAULT_SATURATION,
        indoor_cameras: frozenset[str] | None = None,
        update_interval: float = 30.0,
    ) -> None:
        self._half_life = half_life
        self._saturation = saturation
        self._indoor_cameras = indoor_cameras or frozenset()
        self._update_interval = update_interval

        self._events: list[StressEvent] = []
        self._sustained_ema: float = 0.0
        self._last_score: dict | None = None

        # Daily tracking
        self._today: DailyStats = DailyStats(date=datetime.now(UTC).date())
        self._daily_history: list[DailyStats] = []

    @property
    def last_score(self) -> dict | None:
        """Most recently computed score, or None if never computed."""
        return self._last_score

    def record_event(
        self,
        group: str,
        trigger_db: float,
        camera: str,
        confidence: float,
        num_cameras: int = 1,
    ) -> None:
        """Record a detection event for stress scoring."""
        self._events.append(
            StressEvent(
                timestamp=time.monotonic(),
                group=group,
                trigger_db=trigger_db,
                camera=camera,
                confidence=confidence,
                num_cameras=num_cameras,
            )
        )

    def compute(self, ambient_data: dict[str, dict] | None = None) -> dict:
        """Compute the composite noise stress score.

        Args:
            ambient_data: Per-camera ambient info dict from stream_manager,
                keyed by camera name. Each value should have 'ema_db' key.

        Returns:
            Dict with score, components, and diagnostics.
        """
        now = time.monotonic()

        # Prune old events
        cutoff = now - EVENT_MAX_AGE
        self._events = [e for e in self._events if e.timestamp > cutoff]

        # Component 1: Ambient
        ambient_component = self._compute_ambient(ambient_data)

        # Component 2: Events
        event_component = self._compute_events(now)

        # Component 3: Sustained (EMA of event component)
        sustained_component = self._compute_sustained(event_component)

        # Composite score — ADHD/autism profile weights sustained heavily
        raw = (
            0.20 * ambient_component
            + 0.45 * event_component
            + 0.35 * sustained_component
        )
        score = max(0.0, min(100.0, raw))

        # Diagnostics
        top_stressor = self._get_top_stressor(now)
        dominant_camera = self._get_dominant_camera(now)
        active_high = self._has_active_high_events(now)
        recent_count = sum(1 for e in self._events if now - e.timestamp < self._half_life * 2)

        # Daily tracking — roll over at midnight UTC
        self._update_daily(score, top_stressor)

        result = {
            "score": round(score, 1),
            "ambient_component": round(ambient_component, 1),
            "event_component": round(event_component, 1),
            "sustained_component": round(sustained_component, 1),
            "recent_event_count": recent_count,
            "top_stressor": top_stressor,
            "dominant_camera": dominant_camera,
            "active_high_stress": active_high,
            "daily_avg": self._today.avg_score,
            "daily_min": round(self._today.min_score, 1) if self._today.count > 0 else 0.0,
            "daily_max": round(self._today.max_score, 1),
            "daily_samples": self._today.count,
        }
        self._last_score = result
        return result

    def _compute_ambient(self, ambient_data: dict[str, dict] | None) -> float:
        """Compute ambient component (0-100) from per-camera EMA dB."""
        if not ambient_data:
            return 0.0

        max_normalized = 0.0
        db_range = LOUD_CEILING_DB - QUIET_FLOOR_DB
        if db_range == 0:
            return 0.0

        for camera, info in ambient_data.items():
            ema_db = info.get("ema_db")
            if ema_db is None:
                continue

            normalized = max(0.0, min(1.0, (ema_db - QUIET_FLOOR_DB) / db_range))

            # Indoor multiplier
            if camera in self._indoor_cameras:
                normalized *= INDOOR_MULTIPLIER

            max_normalized = max(max_normalized, normalized)

        return min(100.0, max_normalized * 100.0)

    def _compute_events(self, now: float) -> float:
        """Compute event component (0-100) from decayed impulse sum.

        Each impulse = tier_weight * loudness * confidence * camera_factor.
        Indoor multiplier is only applied in the ambient component to avoid
        double-weighting indoor cameras.

        Overload cascade (ADHD/autism): when sustained EMA > 50, all positive
        sounds hit harder (up to 2x at sustained=100). Above sustained=85,
        masking/calming sounds lose effectiveness entirely.
        """
        if not self._events:
            return 0.0

        has_active_high = self._has_active_high_events(now)

        # Overload cascade: amplify all sounds when already stressed
        extreme_overload = self._sustained_ema > EXTREME_OVERLOAD
        overload_multiplier = 1.0 + max(
            0.0, (self._sustained_ema - OVERLOAD_ONSET) / OVERLOAD_RANGE
        )
        overload_multiplier = min(overload_multiplier, 2.0)

        total = 0.0

        for event in self._events:
            weight = get_tier_weight(event.group)

            if weight < 0:
                # Calming: skip when HIGH active OR during extreme overload
                if has_active_high or extreme_overload:
                    continue
            else:
                # Positive stressor: amplify during overload
                weight *= overload_multiplier

            # Loudness factor
            loudness = self._loudness_factor(event.trigger_db)

            # Confidence factor — higher-confidence detections contribute more
            confidence = max(0.1, min(1.0, event.confidence))

            # Camera factor
            camera_factor = CAMERA_FACTOR_BASE + CAMERA_FACTOR_PER_EXTRA * max(0, event.num_cameras - 1)

            impulse = weight * loudness * confidence * camera_factor

            # Exponential decay
            age = now - event.timestamp
            decayed = impulse * (0.5 ** (age / self._half_life))

            total += decayed

        # Saturation curve: 100 * (1 - e^(-total / saturation))
        if total <= 0:
            return 0.0
        return 100.0 * (1.0 - math.exp(-total / self._saturation))

    def _compute_sustained(self, event_component: float) -> float:
        """Update and return the sustained EMA component.

        ADHD/autism profile: moderate attack (~6 min to saturate),
        very slow release (~90 min to fully decay). Reflects sensory
        sensitization — once overloaded, recovery is much slower than
        onset. Repeated bursts accumulate even if individual events decay.
        """
        # moderate attack (~6 min to saturate) vs very slow release (~90 min to decay)
        alpha = 0.08 if event_component > self._sustained_ema else 0.005

        self._sustained_ema += alpha * (event_component - self._sustained_ema)
        return self._sustained_ema

    @staticmethod
    def _loudness_factor(trigger_db: float) -> float:
        """Map trigger dB to [0.3, 2.0] range."""
        db_range = LOUDNESS_DB_MAX - LOUDNESS_DB_MIN
        if db_range == 0:
            return 1.0
        normalized = max(0.0, min(1.0, (trigger_db - LOUDNESS_DB_MIN) / db_range))
        return LOUDNESS_FACTOR_MIN + normalized * (LOUDNESS_FACTOR_MAX - LOUDNESS_FACTOR_MIN)

    def _has_active_high_events(self, now: float) -> bool:
        """Check if any HIGH-tier events are active (within 2 half-lives)."""
        threshold = self._half_life * 2
        return any(
            e.group in _HIGH_TIER_GROUPS and (now - e.timestamp) < threshold
            for e in self._events
        )

    def _get_top_stressor(self, now: float) -> str | None:
        """Get the group contributing most stress currently."""
        if not self._events:
            return None

        group_stress: dict[str, float] = {}
        for event in self._events:
            weight = get_tier_weight(event.group)
            if weight < 0:
                continue
            loudness = self._loudness_factor(event.trigger_db)
            age = now - event.timestamp
            decayed = weight * loudness * (0.5 ** (age / self._half_life))
            group_stress[event.group] = group_stress.get(event.group, 0.0) + decayed

        if not group_stress:
            return None
        return max(group_stress, key=lambda g: group_stress[g])

    def _get_dominant_camera(self, now: float) -> str | None:
        """Get the camera contributing most events recently."""
        if not self._events:
            return None

        camera_counts: dict[str, int] = {}
        threshold = self._half_life * 2
        for event in self._events:
            if now - event.timestamp < threshold:
                camera_counts[event.camera] = camera_counts.get(event.camera, 0) + 1

        if not camera_counts:
            return None
        return max(camera_counts, key=lambda c: camera_counts[c])

    def _update_daily(self, score: float, top_stressor: str | None) -> None:
        """Update daily stats, rolling over at midnight UTC."""
        today = datetime.now(UTC).date()
        if self._today.date != today:
            # Roll over: finalize yesterday and start a new day
            if self._today.count > 0:
                self._daily_history.append(self._today)
                # Keep only the last DAILY_HISTORY_MAX days
                if len(self._daily_history) > DAILY_HISTORY_MAX:
                    self._daily_history = self._daily_history[-DAILY_HISTORY_MAX:]
            self._today = DailyStats(date=today)

        # Compute weight for top_stressor tracking
        stressor_weight = get_tier_weight(top_stressor) if top_stressor else 0.0
        self._today.record(score, top_stressor, stressor_weight)

    @property
    def daily_history(self) -> list[dict]:
        """Return the rolling daily history as a list of dicts."""
        result = [d.to_dict() for d in self._daily_history]
        if self._today.count > 0:
            result.append(self._today.to_dict())
        return result

    def status(self) -> dict:
        """Return status info for the /status endpoint."""
        result: dict = {
            "enabled": True,
            "event_buffer_size": len(self._events),
            "sustained_ema": round(self._sustained_ema, 1),
        }
        if self._last_score is not None:
            result["last_score"] = self._last_score
        if self._daily_history or self._today.count > 0:
            result["daily_history"] = self.daily_history
        return result
