"""Tests for AudioSet label grouping."""

import pytest

from src.labels import (
    LABEL_GROUPS,
    AudioSetLabels,
    get_group_for_label,
    get_top_group_match,
)


class TestAudioSetLabels:
    """Test the AudioSet label index mapping."""

    def test_loads_527_labels(self):
        labels = AudioSetLabels()
        assert len(labels) == 527

    def test_get_label_by_index(self):
        labels = AudioSetLabels()
        assert labels[0] == "Speech"
        assert labels[3] == "Child speech, kid speaking"
        assert labels[74] == "Dog"

    def test_get_index_by_label(self):
        labels = AudioSetLabels()
        assert labels.index("Speech") == 0
        assert labels.index("Dog") == 74

    def test_index_raises_for_unknown(self):
        labels = AudioSetLabels()
        with pytest.raises(ValueError, match="not found"):
            labels.index("NotARealLabel")

    def test_contains(self):
        labels = AudioSetLabels()
        assert "Speech" in labels
        assert "Dog" in labels
        assert "FakeLabel" not in labels


class TestLabelGroups:
    """Test the label-to-group mapping."""

    def test_all_groups_exist(self):
        expected = {
            # Safety & security
            "smoke_alarm",
            "glass_break",
            "siren",
            "gunshot_explosion",
            "screaming",
            # People & pets
            "dog_bark",
            "cat_meow",
            "crying",
            "speech",
            "cough_sneeze",
            "footsteps",
            # Doors & entry
            "doorbell",
            "knock",
            "door",
            "cabinet",
            # Environment
            "wind",
            "rain_storm",
            "music",
            "vehicle",
            "car_horn",
            "aircraft",
            # Household
            "vacuum_cleaner",
            "water_running",
            "kitchen_appliance",
            "power_tools",
            "alarm_beep",
            # Equipment monitoring
            "hvac_mechanical",
            "mechanical_anomaly",
            "water_leak",
            "electrical_anomaly",
            # Media (TV/movie audio confuser)
            "media",
        }
        assert set(LABEL_GROUPS.keys()) == expected

    def test_dog_bark_labels(self):
        assert "Dog" in LABEL_GROUPS["dog_bark"]
        assert "Bark" in LABEL_GROUPS["dog_bark"]
        assert "Bow-wow" in LABEL_GROUPS["dog_bark"]
        assert "Howl" in LABEL_GROUPS["dog_bark"]

    def test_cat_meow_labels(self):
        assert "Cat" in LABEL_GROUPS["cat_meow"]
        assert "Meow" in LABEL_GROUPS["cat_meow"]
        assert "Purr" in LABEL_GROUPS["cat_meow"]

    def test_crying_labels(self):
        assert "Baby cry, infant cry" in LABEL_GROUPS["crying"]
        assert "Crying, sobbing" in LABEL_GROUPS["crying"]
        assert "Wail, moan" in LABEL_GROUPS["crying"]

    def test_smoke_alarm_labels(self):
        assert "Smoke detector, smoke alarm" in LABEL_GROUPS["smoke_alarm"]
        assert "Fire alarm" in LABEL_GROUPS["smoke_alarm"]

    def test_glass_break_labels(self):
        assert "Shatter" in LABEL_GROUPS["glass_break"]
        assert "Smash, crash" in LABEL_GROUPS["glass_break"]
        assert "Breaking" in LABEL_GROUPS["glass_break"]
        assert "Glass" not in LABEL_GROUPS["glass_break"]

    def test_siren_labels(self):
        assert "Siren" in LABEL_GROUPS["siren"]
        assert "Civil defense siren" in LABEL_GROUPS["siren"]

    def test_knock_labels(self):
        assert "Knock" in LABEL_GROUPS["knock"]
        assert "Tap" in LABEL_GROUPS["knock"]

    def test_gunshot_explosion_labels(self):
        assert "Gunshot, gunfire" in LABEL_GROUPS["gunshot_explosion"]
        assert "Explosion" in LABEL_GROUPS["gunshot_explosion"]
        assert "Fireworks" in LABEL_GROUPS["gunshot_explosion"]

    def test_screaming_labels(self):
        assert "Screaming" in LABEL_GROUPS["screaming"]
        assert "Shout" in LABEL_GROUPS["screaming"]

    def test_cough_sneeze_labels(self):
        assert "Cough" in LABEL_GROUPS["cough_sneeze"]
        assert "Sneeze" in LABEL_GROUPS["cough_sneeze"]

    def test_footsteps_labels(self):
        assert "Walk, footsteps" in LABEL_GROUPS["footsteps"]
        assert "Run" in LABEL_GROUPS["footsteps"]

    def test_door_labels(self):
        assert "Door" in LABEL_GROUPS["door"]
        assert "Slam" in LABEL_GROUPS["door"]
        assert "Sliding door" in LABEL_GROUPS["door"]
        assert "Cupboard open or close" not in LABEL_GROUPS["door"]

    def test_cabinet_labels(self):
        assert "Cupboard open or close" in LABEL_GROUPS["cabinet"]
        assert "Drawer open or close" in LABEL_GROUPS["cabinet"]

    def test_rain_storm_labels(self):
        assert "Rain" in LABEL_GROUPS["rain_storm"]
        assert "Thunderstorm" in LABEL_GROUPS["rain_storm"]
        assert "Thunder" in LABEL_GROUPS["rain_storm"]

    def test_music_labels(self):
        assert "Music" in LABEL_GROUPS["music"]
        assert "Rock music" in LABEL_GROUPS["music"]

    def test_vehicle_labels(self):
        assert "Vehicle" in LABEL_GROUPS["vehicle"]
        assert "Car" in LABEL_GROUPS["vehicle"]
        assert "Truck" in LABEL_GROUPS["vehicle"]

    def test_car_horn_labels(self):
        assert "Vehicle horn, car horn, honking" in LABEL_GROUPS["car_horn"]
        assert "Car alarm" in LABEL_GROUPS["car_horn"]

    def test_vacuum_cleaner_labels(self):
        assert "Vacuum cleaner" in LABEL_GROUPS["vacuum_cleaner"]

    def test_water_running_labels(self):
        assert "Water tap, faucet" in LABEL_GROUPS["water_running"]
        assert "Toilet flush" in LABEL_GROUPS["water_running"]

    def test_kitchen_appliance_labels(self):
        assert "Microwave oven" in LABEL_GROUPS["kitchen_appliance"]
        assert "Blender" in LABEL_GROUPS["kitchen_appliance"]
        assert "Dishes, pots, and pans" in LABEL_GROUPS["kitchen_appliance"]

    def test_power_tools_labels(self):
        assert "Power tool" in LABEL_GROUPS["power_tools"]
        assert "Chainsaw" in LABEL_GROUPS["power_tools"]
        assert "Lawn mower" in LABEL_GROUPS["power_tools"]

    def test_alarm_beep_labels(self):
        assert "Alarm" in LABEL_GROUPS["alarm_beep"]
        assert "Buzzer" in LABEL_GROUPS["alarm_beep"]
        assert "Beep, bleep" in LABEL_GROUPS["alarm_beep"]

    def test_hvac_mechanical_labels(self):
        assert "Mechanical fan" in LABEL_GROUPS["hvac_mechanical"]
        assert "Air conditioning" in LABEL_GROUPS["hvac_mechanical"]

    def test_mechanical_anomaly_labels(self):
        assert "Engine knocking" in LABEL_GROUPS["mechanical_anomaly"]
        assert "Squeal" in LABEL_GROUPS["mechanical_anomaly"]
        assert "Vibration" in LABEL_GROUPS["mechanical_anomaly"]
        assert "Creak" in LABEL_GROUPS["mechanical_anomaly"]

    def test_water_leak_labels(self):
        assert "Drip" in LABEL_GROUPS["water_leak"]
        assert "Trickle, dribble" in LABEL_GROUPS["water_leak"]

    def test_electrical_anomaly_labels(self):
        assert "Buzz" in LABEL_GROUPS["electrical_anomaly"]
        assert "Mains hum" in LABEL_GROUPS["electrical_anomaly"]

    def test_all_labels_are_valid_audioset(self):
        """Every label in every group must be a real AudioSet label."""
        audioset = AudioSetLabels()
        for group, labels in LABEL_GROUPS.items():
            for label in labels:
                assert (
                    label in audioset
                ), f"{label!r} in group {group!r} is not a valid AudioSet label"


class TestGetGroupForLabel:
    """Test single-label group lookup."""

    def test_dog_maps_to_dog_bark(self):
        assert get_group_for_label("Dog") == "dog_bark"

    def test_bark_maps_to_dog_bark(self):
        assert get_group_for_label("Bark") == "dog_bark"

    def test_cat_maps_to_cat_meow(self):
        assert get_group_for_label("Cat") == "cat_meow"

    def test_unknown_returns_none(self):
        assert get_group_for_label("Accordion") is None


class TestGetTopGroupMatch:
    """Test finding best match from a list of (label, score) pairs."""

    def test_single_dog_label(self):
        predictions = [("Dog", 0.8), ("Speech", 0.1)]
        result = get_top_group_match(predictions, threshold=0.1)
        assert result is not None
        group, confidence, raw_label = result
        assert group == "dog_bark"
        assert confidence == 0.8
        assert raw_label == "Dog"

    def test_multiple_dog_labels_takes_highest(self):
        predictions = [("Dog", 0.4), ("Bark", 0.7), ("Howl", 0.2)]
        result = get_top_group_match(predictions, threshold=0.1)
        assert result is not None
        group, confidence, raw_label = result
        assert group == "dog_bark"
        assert confidence == 0.7
        assert raw_label == "Bark"

    def test_below_threshold_returns_none(self):
        predictions = [("Dog", 0.05)]
        result = get_top_group_match(predictions, threshold=0.1)
        assert result is None

    def test_no_matching_groups(self):
        predictions = [("Accordion", 0.9)]
        result = get_top_group_match(predictions, threshold=0.1)
        assert result is None

    def test_returns_all_matching_groups(self):
        predictions = [("Dog", 0.8), ("Cat", 0.6), ("Speech", 0.3)]
        results = get_top_group_match(predictions, threshold=0.1, all_groups=True)
        assert len(results) == 3
        groups = {r[0] for r in results}
        assert groups == {"dog_bark", "cat_meow", "speech"}

    def test_all_groups_sorted_by_confidence(self):
        predictions = [("Dog", 0.3), ("Cat", 0.8), ("Speech", 0.5)]
        results = get_top_group_match(predictions, threshold=0.1, all_groups=True)
        assert results[0][0] == "cat_meow"
        assert results[1][0] == "speech"
        assert results[2][0] == "dog_bark"

    def test_empty_predictions(self):
        result = get_top_group_match([], threshold=0.1)
        assert result is None


class TestGetTopGroupMatchPerGroupThreshold:
    """Test per-group threshold overrides."""

    def test_per_group_threshold_filters_low_confidence(self):
        predictions = [("Dog", 0.4), ("Music", 0.3)]
        group_thresholds = {"music": 0.60}
        result = get_top_group_match(
            predictions, threshold=0.15, all_groups=True,
            group_thresholds=group_thresholds,
        )
        groups = {r[0] for r in result}
        assert "dog_bark" in groups
        assert "music" not in groups

    def test_per_group_threshold_allows_high_confidence(self):
        predictions = [("Music", 0.75)]
        group_thresholds = {"music": 0.60}
        result = get_top_group_match(
            predictions, threshold=0.15, all_groups=True,
            group_thresholds=group_thresholds,
        )
        assert len(result) == 1
        assert result[0][0] == "music"

    def test_disabled_group_excluded(self):
        predictions = [("Dog", 0.8), ("Vehicle horn, car horn, honking", 0.9)]
        result = get_top_group_match(
            predictions, threshold=0.15, all_groups=True,
            disabled_groups={"car_horn"},
        )
        groups = {r[0] for r in result}
        assert "dog_bark" in groups
        assert "car_horn" not in groups

    def test_no_per_group_thresholds_uses_global(self):
        predictions = [("Dog", 0.3), ("Music", 0.3)]
        result = get_top_group_match(
            predictions, threshold=0.15, all_groups=True,
        )
        assert len(result) == 2

    def test_single_result_mode_respects_per_group(self):
        predictions = [("Music", 0.5), ("Dog", 0.3)]
        group_thresholds = {"music": 0.60}
        result = get_top_group_match(
            predictions, threshold=0.15,
            group_thresholds=group_thresholds,
        )
        assert result is not None
        assert result[0] == "dog_bark"
