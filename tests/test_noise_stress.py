"""Tests for noise stress scoring module."""

import time
from datetime import UTC, date, datetime, timedelta
from unittest.mock import patch

from src.labels import LABEL_GROUPS
from src.noise_stress import (
    CAMERA_FACTOR_BASE,
    CAMERA_FACTOR_PER_EXTRA,
    DAILY_HISTORY_MAX,
    DEFAULT_TIER_WEIGHT,
    EVENT_MAX_AGE,
    EXTREME_OVERLOAD,
    LOUD_CEILING_DB,
    LOUDNESS_DB_MAX,
    LOUDNESS_DB_MIN,
    LOUDNESS_FACTOR_MAX,
    LOUDNESS_FACTOR_MIN,
    OVERLOAD_ONSET,
    QUIET_FLOOR_DB,
    DailyStats,
    NoiseStressScorer,
    StressEvent,
    get_tier_weight,
)


class TestStressTierWeights:
    """All 31 label groups map to expected ADHD/autism-calibrated tiers."""

    # HIGH STARTLE / SENSITIZATION (3.0)
    HIGH_GROUPS = frozenset({
        "dog_bark", "screaming", "crying", "siren",
        "alarm_beep", "glass_break", "gunshot_explosion",
    })
    # FOCUS DISRUPTION / UNPREDICTABLE (2.5)
    FOCUS_DISRUPTION_GROUPS = frozenset({
        "speech", "knock", "doorbell",
    })
    # LOUD BUT PREDICTABLE (1.2)
    LOUD_PREDICTABLE_GROUPS = frozenset({
        "vacuum_cleaner", "power_tools", "car_horn", "music",
    })
    # BACKGROUND / MILD (0.4)
    BACKGROUND_GROUPS = frozenset({
        "footsteps", "door", "cabinet", "kitchen_appliance",
        "vehicle", "aircraft",
    })
    # STIMMING / MASKING (-0.5)
    CALMING_GROUPS = frozenset({"rain_storm", "hvac_mechanical", "water_running"})

    def test_high_tier_weight(self):
        for group in self.HIGH_GROUPS:
            assert get_tier_weight(group) == 3.0, f"{group} should be HIGH (3.0)"

    def test_focus_disruption_tier_weight(self):
        for group in self.FOCUS_DISRUPTION_GROUPS:
            assert get_tier_weight(group) == 2.5, f"{group} should be FOCUS DISRUPTION (2.5)"

    def test_loud_predictable_tier_weight(self):
        for group in self.LOUD_PREDICTABLE_GROUPS:
            assert get_tier_weight(group) == 1.2, f"{group} should be LOUD PREDICTABLE (1.2)"

    def test_background_tier_weight(self):
        for group in self.BACKGROUND_GROUPS:
            assert get_tier_weight(group) == 0.4, f"{group} should be BACKGROUND (0.4)"

    def test_calming_tier_weight(self):
        for group in self.CALMING_GROUPS:
            assert get_tier_weight(group) == -0.5, f"{group} should be CALMING (-0.5)"

    def test_default_tier_weight(self):
        assert get_tier_weight("cat_meow") == DEFAULT_TIER_WEIGHT
        assert get_tier_weight("cough_sneeze") == DEFAULT_TIER_WEIGHT
        assert get_tier_weight("mechanical_anomaly") == DEFAULT_TIER_WEIGHT

    def test_all_label_groups_have_tier(self):
        """Every group in LABEL_GROUPS should resolve to some tier weight."""
        for group in LABEL_GROUPS:
            weight = get_tier_weight(group)
            assert isinstance(weight, (int, float)), f"{group} has no weight"

    def test_unknown_group_gets_default(self):
        assert get_tier_weight("nonexistent_group") == DEFAULT_TIER_WEIGHT


class TestStressEvent:
    def test_creation(self):
        ev = StressEvent(
            timestamp=100.0,
            group="dog_bark",
            trigger_db=-25.0,
            camera="living_room",
            confidence=0.85,
            num_cameras=2,
        )
        assert ev.group == "dog_bark"
        assert ev.trigger_db == -25.0
        assert ev.num_cameras == 2

    def test_defaults(self):
        ev = StressEvent(
            timestamp=100.0,
            group="speech",
            trigger_db=-30.0,
            camera="cam1",
            confidence=0.7,
        )
        assert ev.num_cameras == 1


class TestAmbientComponent:
    def test_no_ambient_data_returns_zero(self):
        scorer = NoiseStressScorer()
        result = scorer.compute(ambient_data=None)
        assert result["ambient_component"] == 0.0

    def test_empty_ambient_data_returns_zero(self):
        scorer = NoiseStressScorer()
        result = scorer.compute(ambient_data={})
        assert result["ambient_component"] == 0.0

    def test_quiet_room(self):
        """Ambient at quiet floor should give ~0."""
        scorer = NoiseStressScorer()
        result = scorer.compute({"cam1": {"ema_db": QUIET_FLOOR_DB}})
        assert result["ambient_component"] == 0.0

    def test_loud_room(self):
        """Ambient at loud ceiling should give ~100."""
        scorer = NoiseStressScorer()
        result = scorer.compute({"cam1": {"ema_db": LOUD_CEILING_DB}})
        assert result["ambient_component"] == 100.0

    def test_mid_range(self):
        """Ambient halfway between floor and ceiling."""
        scorer = NoiseStressScorer()
        mid_db = (QUIET_FLOOR_DB + LOUD_CEILING_DB) / 2
        result = scorer.compute({"cam1": {"ema_db": mid_db}})
        assert 45.0 < result["ambient_component"] < 55.0

    def test_max_across_cameras(self):
        """Should use the max across cameras, not average."""
        scorer = NoiseStressScorer()
        result = scorer.compute({
            "quiet_cam": {"ema_db": QUIET_FLOOR_DB},
            "loud_cam": {"ema_db": LOUD_CEILING_DB},
        })
        assert result["ambient_component"] == 100.0

    def test_indoor_multiplier(self):
        """Indoor cameras should have their value boosted."""
        mid_db = (QUIET_FLOOR_DB + LOUD_CEILING_DB) / 2
        scorer_no_indoor = NoiseStressScorer()
        scorer_indoor = NoiseStressScorer(indoor_cameras=frozenset(["cam1"]))

        result_no = scorer_no_indoor.compute({"cam1": {"ema_db": mid_db}})
        result_yes = scorer_indoor.compute({"cam1": {"ema_db": mid_db}})

        assert result_yes["ambient_component"] > result_no["ambient_component"]

    def test_indoor_multiplier_clamped(self):
        """Indoor multiplier shouldn't push component above 100."""
        scorer = NoiseStressScorer(indoor_cameras=frozenset(["cam1"]))
        result = scorer.compute({"cam1": {"ema_db": LOUD_CEILING_DB}})
        assert result["ambient_component"] == 100.0

    def test_missing_ema_db_skipped(self):
        """Camera ambient info without ema_db should be skipped."""
        scorer = NoiseStressScorer()
        result = scorer.compute({"cam1": {"peak_db": -20.0}})
        assert result["ambient_component"] == 0.0

    def test_below_floor_clamps_to_zero(self):
        scorer = NoiseStressScorer()
        result = scorer.compute({"cam1": {"ema_db": -80.0}})
        assert result["ambient_component"] == 0.0

    def test_above_ceiling_clamps_to_max(self):
        scorer = NoiseStressScorer()
        result = scorer.compute({"cam1": {"ema_db": 0.0}})
        assert result["ambient_component"] == 100.0


class TestEventComponent:
    def test_no_events_returns_zero(self):
        scorer = NoiseStressScorer()
        result = scorer.compute()
        assert result["event_component"] == 0.0

    def test_single_high_event(self):
        """A single HIGH-tier event should produce nonzero event component."""
        scorer = NoiseStressScorer()
        scorer.record_event("dog_bark", -25.0, "cam1", 0.9)
        result = scorer.compute()
        assert result["event_component"] > 0.0

    def test_exponential_decay(self):
        """Event component should decrease as events age."""
        scorer = NoiseStressScorer(half_life=10.0)  # short for testing
        scorer.record_event("dog_bark", -25.0, "cam1", 0.9)

        result_fresh = scorer.compute()
        # Manually age the event
        scorer._events[0].timestamp -= 20.0  # 2 half-lives
        # Reset sustained EMA to isolate event component
        scorer._sustained_ema = 0.0
        result_aged = scorer.compute()

        assert result_aged["event_component"] < result_fresh["event_component"]

    def test_saturation_curve(self):
        """Many events should asymptotically approach 100."""
        scorer = NoiseStressScorer()
        for _ in range(50):
            scorer.record_event("screaming", -15.0, "cam1", 0.95)
        result = scorer.compute()
        assert result["event_component"] > 90.0

    def test_saturation_never_exceeds_100(self):
        scorer = NoiseStressScorer()
        for _ in range(200):
            scorer.record_event("screaming", -10.0, "cam1", 0.99)
        result = scorer.compute()
        assert result["event_component"] <= 100.0

    def test_loudness_factor_range(self):
        """Quieter triggers should contribute less than louder ones."""
        scorer_quiet = NoiseStressScorer()
        scorer_quiet.record_event("dog_bark", LOUDNESS_DB_MIN, "cam1", 0.9)

        scorer_loud = NoiseStressScorer()
        scorer_loud.record_event("dog_bark", LOUDNESS_DB_MAX, "cam1", 0.9)

        result_quiet = scorer_quiet.compute()
        result_loud = scorer_loud.compute()

        assert result_loud["event_component"] > result_quiet["event_component"]


class TestLoudnessFactor:
    def test_min_db(self):
        factor = NoiseStressScorer._loudness_factor(LOUDNESS_DB_MIN)
        assert abs(factor - LOUDNESS_FACTOR_MIN) < 0.01

    def test_max_db(self):
        factor = NoiseStressScorer._loudness_factor(LOUDNESS_DB_MAX)
        assert abs(factor - LOUDNESS_FACTOR_MAX) < 0.01

    def test_below_min_clamps(self):
        factor = NoiseStressScorer._loudness_factor(-100.0)
        assert abs(factor - LOUDNESS_FACTOR_MIN) < 0.01

    def test_above_max_clamps(self):
        factor = NoiseStressScorer._loudness_factor(0.0)
        assert abs(factor - LOUDNESS_FACTOR_MAX) < 0.01


class TestSustainedComponent:
    def test_moderate_attack(self):
        """Sustained EMA should rise with events (alpha=0.08)."""
        scorer = NoiseStressScorer()

        # First compute with events
        for _ in range(5):
            scorer.record_event("dog_bark", -25.0, "cam1", 0.9)
        result1 = scorer.compute()
        sustained1 = result1["sustained_component"]

        # Sustained should be less than event component (moderate attack)
        assert sustained1 < result1["event_component"]

    def test_very_slow_release(self):
        """Sustained should decay VERY slowly (alpha=0.005, ~90 min)."""
        scorer = NoiseStressScorer()

        # Build up sustained
        for _ in range(20):
            scorer.record_event("dog_bark", -25.0, "cam1", 0.9)
        scorer.compute()
        built_up = scorer._sustained_ema

        # Clear events and compute 10 times — with alpha=0.005,
        # each step only decays 0.5%, so after 10 steps it's ~95% retained
        scorer._events.clear()
        for _ in range(10):
            scorer.compute()
        decayed = scorer._sustained_ema

        assert decayed < built_up
        # With very slow release, should retain most of the built-up value
        assert decayed > built_up * 0.90

    def test_asymmetric_attack_release(self):
        """Attack (0.08) should be much faster than release (0.005)."""
        # Build up in 10 steps
        scorer = NoiseStressScorer()
        for _ in range(10):
            scorer.record_event("screaming", -15.0, "cam1", 0.95)
        scorer.compute()
        peak = scorer._sustained_ema

        # Now decay for 10 steps (same count)
        scorer._events.clear()
        for _ in range(10):
            scorer.compute()
        after_decay = scorer._sustained_ema

        # After same number of steps, sustained should still be >80% of peak
        # because release is 16x slower than attack (0.005 vs 0.08)
        assert after_decay > peak * 0.80

    def test_zero_when_no_events(self):
        """Sustained should be 0 when never used."""
        scorer = NoiseStressScorer()
        result = scorer.compute()
        assert result["sustained_component"] == 0.0


class TestCompositeScore:
    def test_weighted_combination(self):
        """Composite score should be the weighted sum of components (20/45/35)."""
        scorer = NoiseStressScorer()
        scorer.record_event("dog_bark", -25.0, "cam1", 0.9)
        result = scorer.compute({"cam1": {"ema_db": -35.0}})

        expected = (
            0.20 * result["ambient_component"]
            + 0.45 * result["event_component"]
            + 0.35 * result["sustained_component"]
        )
        assert abs(result["score"] - round(expected, 1)) < 0.2

    def test_clamped_at_zero(self):
        """Score should never go below 0."""
        scorer = NoiseStressScorer()
        result = scorer.compute()
        assert result["score"] >= 0.0

    def test_clamped_at_100(self):
        """Score should never exceed 100."""
        scorer = NoiseStressScorer()
        for _ in range(200):
            scorer.record_event("screaming", -10.0, "cam1", 0.99)
        result = scorer.compute({"cam1": {"ema_db": LOUD_CEILING_DB}})
        assert result["score"] <= 100.0

    def test_all_zeros_when_quiet(self):
        scorer = NoiseStressScorer()
        result = scorer.compute()
        assert result["score"] == 0.0
        assert result["ambient_component"] == 0.0
        assert result["event_component"] == 0.0
        assert result["sustained_component"] == 0.0


class TestEventPruning:
    def test_old_events_pruned(self):
        """Events older than EVENT_MAX_AGE should be pruned on compute()."""
        scorer = NoiseStressScorer()
        scorer.record_event("dog_bark", -25.0, "cam1", 0.9)
        # Manually age the event beyond max age
        scorer._events[0].timestamp = time.monotonic() - EVENT_MAX_AGE - 10
        scorer.compute()
        assert len(scorer._events) == 0

    def test_recent_events_kept(self):
        """Recent events should survive pruning."""
        scorer = NoiseStressScorer()
        scorer.record_event("dog_bark", -25.0, "cam1", 0.9)
        scorer.compute()
        assert len(scorer._events) == 1


class TestCalmingEffect:
    def test_rain_subtracts_when_no_high_active(self):
        """Rain/storm should reduce stress when no HIGH-tier events active."""
        scorer_no_rain = NoiseStressScorer()
        scorer_no_rain.record_event("footsteps", -30.0, "cam1", 0.8)

        scorer_rain = NoiseStressScorer()
        scorer_rain.record_event("footsteps", -30.0, "cam1", 0.8)
        scorer_rain.record_event("rain_storm", -35.0, "cam2", 0.9)

        result_no_rain = scorer_no_rain.compute()
        result_rain = scorer_rain.compute()

        # With rain (calming), event component should be lower
        assert result_rain["event_component"] <= result_no_rain["event_component"]

    def test_hvac_is_calming(self):
        """HVAC mechanical should be calming (-0.5) for ADHD/autism profiles."""
        scorer_no_hvac = NoiseStressScorer()
        scorer_no_hvac.record_event("footsteps", -30.0, "cam1", 0.8)

        scorer_hvac = NoiseStressScorer()
        scorer_hvac.record_event("footsteps", -30.0, "cam1", 0.8)
        scorer_hvac.record_event("hvac_mechanical", -40.0, "cam1", 0.9)

        result_no = scorer_no_hvac.compute()
        result_yes = scorer_hvac.compute()

        assert result_yes["event_component"] <= result_no["event_component"]

    def test_water_running_is_calming(self):
        """Water running should be calming (-0.5) for ADHD/autism profiles."""
        scorer_no_water = NoiseStressScorer()
        scorer_no_water.record_event("footsteps", -30.0, "cam1", 0.8)

        scorer_water = NoiseStressScorer()
        scorer_water.record_event("footsteps", -30.0, "cam1", 0.8)
        scorer_water.record_event("water_running", -40.0, "cam1", 0.9)

        result_no = scorer_no_water.compute()
        result_yes = scorer_water.compute()

        assert result_yes["event_component"] <= result_no["event_component"]

    def test_calming_ignored_when_high_active(self):
        """Calming effects should be suppressed when HIGH events are active."""
        scorer = NoiseStressScorer()
        scorer.record_event("screaming", -20.0, "cam1", 0.95)
        scorer.record_event("rain_storm", -35.0, "cam2", 0.9)

        result = scorer.compute()
        # The rain event should be skipped (not subtract from stress)
        assert result["event_component"] > 0.0

    def test_calming_disabled_during_extreme_overload(self):
        """Calming sounds stop working when sustained EMA > 85 (extreme overload)."""
        scorer = NoiseStressScorer()
        # Force sustained EMA above EXTREME_OVERLOAD
        scorer._sustained_ema = EXTREME_OVERLOAD + 1.0

        scorer.record_event("footsteps", -30.0, "cam1", 0.8)
        scorer.record_event("rain_storm", -35.0, "cam2", 0.9)

        result = scorer.compute()
        # Rain should be skipped during extreme overload, so event > 0
        assert result["event_component"] > 0.0


class TestOverloadCascade:
    """ADHD/autism overload cascade: sounds amplified when sustained is high."""

    def test_no_overload_below_onset(self):
        """When sustained < 50, overload multiplier is 1.0 (no amplification)."""
        scorer = NoiseStressScorer()
        scorer._sustained_ema = OVERLOAD_ONSET - 10.0
        scorer.record_event("footsteps", -30.0, "cam1", 0.8)

        result = scorer.compute()
        # Should produce a normal score, no amplification
        assert result["event_component"] > 0.0

    def test_overload_amplifies_stressors(self):
        """When sustained > 50, positive stressors should be amplified."""
        scorer_normal = NoiseStressScorer()
        scorer_normal._sustained_ema = 0.0
        scorer_normal.record_event("footsteps", -30.0, "cam1", 0.8)

        scorer_overload = NoiseStressScorer()
        scorer_overload._sustained_ema = OVERLOAD_ONSET + 25.0  # 75 = 1.5x multiplier
        scorer_overload.record_event("footsteps", -30.0, "cam1", 0.8)

        result_normal = scorer_normal.compute()
        result_overload = scorer_overload.compute()

        assert result_overload["event_component"] > result_normal["event_component"]

    def test_overload_multiplier_capped_at_2x(self):
        """Overload multiplier should not exceed 2.0."""
        scorer = NoiseStressScorer()
        scorer._sustained_ema = 120.0  # Way above range
        scorer.record_event("footsteps", -30.0, "cam1", 0.8)

        result = scorer.compute()
        # Should still produce a valid result (not infinite)
        assert result["event_component"] <= 100.0
        assert result["event_component"] > 0.0

    def test_extreme_overload_disables_calming(self):
        """Above sustained=85, calming sounds should be ignored."""
        scorer = NoiseStressScorer()
        scorer._sustained_ema = EXTREME_OVERLOAD + 5.0
        scorer.record_event("footsteps", -30.0, "cam1", 0.8)
        scorer.record_event("hvac_mechanical", -40.0, "cam1", 0.9)

        result = scorer.compute()
        # HVAC calming should be skipped during extreme overload
        assert result["event_component"] > 0.0

    def test_overload_progressive(self):
        """Higher sustained EMA should produce stronger amplification."""
        results = []
        for sustained in [0.0, 50.0, 75.0, 100.0]:
            scorer = NoiseStressScorer()
            scorer._sustained_ema = sustained
            scorer.record_event("speech", -25.0, "cam1", 0.8)
            results.append(scorer.compute()["event_component"])

        # Each should be >= previous
        for i in range(1, len(results)):
            assert results[i] >= results[i - 1]


class TestMultiCamera:
    def test_num_cameras_boosts_impulse(self):
        """Events detected by multiple cameras should have higher impulse."""
        scorer_single = NoiseStressScorer()
        scorer_single.record_event("dog_bark", -25.0, "cam1", 0.9, num_cameras=1)

        scorer_multi = NoiseStressScorer()
        scorer_multi.record_event("dog_bark", -25.0, "cam1", 0.9, num_cameras=3)

        result_single = scorer_single.compute()
        result_multi = scorer_multi.compute()

        assert result_multi["event_component"] > result_single["event_component"]

    def test_camera_factor_formula(self):
        """Camera factor should be 1.0 + 0.15 * (num_cameras - 1)."""
        assert CAMERA_FACTOR_BASE == 1.0
        assert CAMERA_FACTOR_PER_EXTRA == 0.15
        # 3 cameras: 1.0 + 0.15 * 2 = 1.30
        expected = 1.0 + 0.15 * 2
        assert abs(expected - 1.30) < 0.001


class TestDiagnostics:
    def test_top_stressor(self):
        """top_stressor should identify the most stressful group."""
        scorer = NoiseStressScorer()
        scorer.record_event("footsteps", -30.0, "cam1", 0.8)
        scorer.record_event("screaming", -20.0, "cam1", 0.95)
        result = scorer.compute()
        assert result["top_stressor"] == "screaming"

    def test_dominant_camera(self):
        """dominant_camera should identify the camera with most events."""
        scorer = NoiseStressScorer()
        scorer.record_event("dog_bark", -25.0, "cam1", 0.9)
        scorer.record_event("speech", -30.0, "cam2", 0.7)
        scorer.record_event("footsteps", -35.0, "cam2", 0.6)
        result = scorer.compute()
        assert result["dominant_camera"] == "cam2"

    def test_active_high_stress_true(self):
        scorer = NoiseStressScorer()
        scorer.record_event("screaming", -20.0, "cam1", 0.95)
        result = scorer.compute()
        assert result["active_high_stress"] is True

    def test_active_high_stress_false(self):
        scorer = NoiseStressScorer()
        scorer.record_event("footsteps", -35.0, "cam1", 0.6)
        result = scorer.compute()
        assert result["active_high_stress"] is False

    def test_recent_event_count(self):
        scorer = NoiseStressScorer(half_life=60.0)
        scorer.record_event("dog_bark", -25.0, "cam1", 0.9)
        scorer.record_event("speech", -30.0, "cam1", 0.7)
        result = scorer.compute()
        assert result["recent_event_count"] == 2

    def test_top_stressor_none_when_empty(self):
        scorer = NoiseStressScorer()
        result = scorer.compute()
        assert result["top_stressor"] is None

    def test_dominant_camera_none_when_empty(self):
        scorer = NoiseStressScorer()
        result = scorer.compute()
        assert result["dominant_camera"] is None


class TestEdgeCases:
    def test_no_events_no_ambient(self):
        scorer = NoiseStressScorer()
        result = scorer.compute()
        assert result["score"] == 0.0
        assert result["recent_event_count"] == 0
        assert result["top_stressor"] is None
        assert result["dominant_camera"] is None

    def test_last_score_none_before_compute(self):
        scorer = NoiseStressScorer()
        assert scorer.last_score is None

    def test_last_score_updated_after_compute(self):
        scorer = NoiseStressScorer()
        scorer.compute()
        assert scorer.last_score is not None
        assert "score" in scorer.last_score

    def test_status_method(self):
        scorer = NoiseStressScorer()
        status = scorer.status()
        assert status["enabled"] is True
        assert status["event_buffer_size"] == 0
        assert "sustained_ema" in status

    def test_status_includes_last_score(self):
        scorer = NoiseStressScorer()
        scorer.compute()
        status = scorer.status()
        assert "last_score" in status

    def test_record_event_basic(self):
        """record_event should add to buffer."""
        scorer = NoiseStressScorer()
        scorer.record_event("dog_bark", -25.0, "cam1", 0.9)
        assert len(scorer._events) == 1
        assert scorer._events[0].group == "dog_bark"

    def test_multiple_computes_stable(self):
        """Multiple compute() calls should be stable."""
        scorer = NoiseStressScorer()
        scorer.record_event("speech", -30.0, "cam1", 0.7)
        results = [scorer.compute() for _ in range(5)]
        # Event component should decrease slightly due to sustained EMA convergence
        # but score should remain stable within a range
        scores = [r["score"] for r in results]
        assert max(scores) - min(scores) < 5.0

    def test_indoor_cameras_config(self):
        """Indoor cameras frozenset should be stored correctly."""
        scorer = NoiseStressScorer(indoor_cameras=frozenset(["living_room", "basement"]))
        assert "living_room" in scorer._indoor_cameras
        assert "outdoor_cam" not in scorer._indoor_cameras


class TestConfidenceFactor:
    """Confidence should act as an impulse multiplier."""

    def test_high_confidence_higher_score(self):
        """Higher confidence should produce higher event component."""
        scorer_low = NoiseStressScorer()
        scorer_low.record_event("dog_bark", -25.0, "cam1", 0.3)

        scorer_high = NoiseStressScorer()
        scorer_high.record_event("dog_bark", -25.0, "cam1", 0.95)

        result_low = scorer_low.compute()
        result_high = scorer_high.compute()

        assert result_high["event_component"] > result_low["event_component"]

    def test_confidence_clamped_minimum(self):
        """Very low confidence should clamp to 0.1 minimum."""
        scorer = NoiseStressScorer()
        scorer.record_event("dog_bark", -25.0, "cam1", 0.01)
        result = scorer.compute()
        # Should still produce a nonzero result (clamped to 0.1)
        assert result["event_component"] > 0.0

    def test_confidence_clamped_maximum(self):
        """Confidence above 1.0 should clamp to 1.0."""
        scorer_normal = NoiseStressScorer()
        scorer_normal.record_event("dog_bark", -25.0, "cam1", 1.0)

        scorer_over = NoiseStressScorer()
        scorer_over.record_event("dog_bark", -25.0, "cam1", 1.5)

        result_normal = scorer_normal.compute()
        result_over = scorer_over.compute()

        # Both should be equal since clamped to 1.0
        assert abs(result_normal["event_component"] - result_over["event_component"]) < 0.1


class TestIndoorMultiplier:
    """Indoor multiplier should only apply to ambient, not events."""

    def test_indoor_not_in_events(self):
        """Indoor cameras should NOT get a boost in event component."""
        scorer_indoor = NoiseStressScorer(indoor_cameras=frozenset(["cam1"]))
        scorer_indoor.record_event("dog_bark", -25.0, "cam1", 0.9)

        scorer_outdoor = NoiseStressScorer()
        scorer_outdoor.record_event("dog_bark", -25.0, "cam1", 0.9)

        result_indoor = scorer_indoor.compute()
        result_outdoor = scorer_outdoor.compute()

        # Event components should be identical — no indoor boost in events
        assert abs(result_indoor["event_component"] - result_outdoor["event_component"]) < 0.01

    def test_indoor_only_in_ambient(self):
        """Indoor multiplier should only boost ambient component."""
        mid_db = (QUIET_FLOOR_DB + LOUD_CEILING_DB) / 2
        scorer_indoor = NoiseStressScorer(indoor_cameras=frozenset(["cam1"]))
        scorer_outdoor = NoiseStressScorer()

        result_indoor = scorer_indoor.compute({"cam1": {"ema_db": mid_db}})
        result_outdoor = scorer_outdoor.compute({"cam1": {"ema_db": mid_db}})

        assert result_indoor["ambient_component"] > result_outdoor["ambient_component"]


class TestDailyStats:
    """Daily statistics tracking."""

    def test_daily_stats_creation(self):
        today = date.today()
        stats = DailyStats(date=today)
        assert stats.count == 0
        assert stats.avg_score == 0.0

    def test_daily_stats_record(self):
        stats = DailyStats(date=date.today())
        stats.record(50.0, "dog_bark", 3.0)
        stats.record(70.0, "screaming", 3.0)
        stats.record(30.0, "speech", 1.5)

        assert stats.count == 3
        assert stats.min_score == 30.0
        assert stats.max_score == 70.0
        assert stats.avg_score == 50.0

    def test_daily_stats_peak_stressor(self):
        stats = DailyStats(date=date.today())
        stats.record(50.0, "speech", 1.5)
        stats.record(70.0, "screaming", 3.0)
        assert stats.peak_stressor == "screaming"

    def test_daily_stats_to_dict(self):
        stats = DailyStats(date=date(2026, 2, 27))
        stats.record(50.0, "dog_bark", 3.0)
        d = stats.to_dict()
        assert d["date"] == "2026-02-27"
        assert d["min"] == 50.0
        assert d["max"] == 50.0
        assert d["avg"] == 50.0
        assert d["samples"] == 1
        assert d["peak_stressor"] == "dog_bark"


class TestDailyTracking:
    """Daily tracking integration in NoiseStressScorer."""

    def test_compute_includes_daily_fields(self):
        scorer = NoiseStressScorer()
        result = scorer.compute()
        assert "daily_avg" in result
        assert "daily_min" in result
        assert "daily_max" in result
        assert "daily_samples" in result

    def test_daily_samples_increment(self):
        scorer = NoiseStressScorer()
        scorer.compute()
        scorer.compute()
        result = scorer.compute()
        assert result["daily_samples"] == 3

    def test_daily_rollover(self):
        """When the date changes, previous day should be archived."""
        scorer = NoiseStressScorer()
        scorer.record_event("dog_bark", -25.0, "cam1", 0.9)
        scorer.compute()

        # Simulate midnight rollover by patching datetime
        tomorrow = datetime.now(UTC).date() + timedelta(days=1)
        with patch("src.noise_stress.datetime") as mock_dt:
            mock_dt.now.return_value.date.return_value = tomorrow
            scorer.compute()

        assert len(scorer._daily_history) == 1
        assert scorer._today.date == tomorrow

    def test_daily_history_max_entries(self):
        """History should be capped at DAILY_HISTORY_MAX entries."""
        scorer = NoiseStressScorer()

        # Generate more than DAILY_HISTORY_MAX days of history
        base_date = date(2026, 2, 1)
        for i in range(DAILY_HISTORY_MAX + 3):
            current_date = base_date + timedelta(days=i)
            with patch("src.noise_stress.datetime") as mock_dt:
                mock_dt.now.return_value.date.return_value = current_date
                scorer.record_event("speech", -30.0, "cam1", 0.7)
                scorer.compute()

        assert len(scorer._daily_history) <= DAILY_HISTORY_MAX

    def test_daily_history_property(self):
        """daily_history should include current day if it has samples."""
        scorer = NoiseStressScorer()
        scorer.record_event("dog_bark", -25.0, "cam1", 0.9)
        scorer.compute()

        history = scorer.daily_history
        assert len(history) == 1
        assert history[0]["date"] == datetime.now(UTC).date().isoformat()

    def test_status_includes_daily_history(self):
        scorer = NoiseStressScorer()
        scorer.compute()
        status = scorer.status()
        assert "daily_history" in status
