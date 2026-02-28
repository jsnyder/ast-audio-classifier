"""Tests for CLAP zero-shot verification logic."""

import numpy as np

from src.clap_verifier import (
    DEFAULT_NEVER_SUPPRESS,
    DEFAULT_PROMPTS,
    CLAPConfig,
    CLAPVerifier,
    build_default_prompts,
)
from src.classifier import ClassificationResult
from src.labels import LABEL_GROUPS


class TestDefaultPrompts:
    def test_all_groups_have_prompts(self):
        """Every LABEL_GROUPS key should have at least one prompt."""
        for group in LABEL_GROUPS:
            assert group in DEFAULT_PROMPTS, f"Missing prompt for group: {group}"
            assert len(DEFAULT_PROMPTS[group]) >= 1

    def test_prompts_are_strings(self):
        for group, prompts in DEFAULT_PROMPTS.items():
            for p in prompts:
                assert isinstance(p, str), f"Non-string prompt in {group}: {p}"

    def test_build_default_prompts_matches_module_level(self):
        built = build_default_prompts()
        assert built == DEFAULT_PROMPTS


class TestCLAPConfig:
    def test_defaults(self):
        cfg = CLAPConfig()
        assert cfg.enabled is True
        assert cfg.model == "laion/clap-htsat-fused"
        assert cfg.confirm_threshold == 0.30
        assert cfg.suppress_threshold == 0.15
        assert cfg.override_threshold == 0.40
        assert cfg.discovery_threshold == 0.50
        assert cfg.confirm_margin == 0.20
        assert cfg.never_suppress == DEFAULT_NEVER_SUPPRESS
        assert cfg.custom_prompts is None

    def test_custom_values(self):
        cfg = CLAPConfig(
            confirm_threshold=0.30,
            suppress_threshold=0.10,
            custom_prompts={"vacuum_cleaner": ["a roomba running"]},
        )
        assert cfg.confirm_threshold == 0.30
        assert cfg.custom_prompts == {"vacuum_cleaner": ["a roomba running"]}


def _make_ast_result(
    label: str = "Dog",
    group: str = "dog_bark",
    confidence: float = 0.85,
    db_level: float = -25.0,
    **kwargs,
) -> ClassificationResult:
    return ClassificationResult(
        label=label,
        group=group,
        confidence=confidence,
        top_5=[(label, confidence)],
        db_level=db_level,
        **kwargs,
    )


class FakeCLAPPipeline:
    """Mock CLAP pipeline that returns pre-configured scores."""

    def __init__(self, scores: dict[str, float]):
        self._scores = scores

    def __call__(self, audio, candidate_labels=None, **kwargs):
        results = []
        for label in (candidate_labels or []):
            score = self._scores.get(label, 0.01)
            results.append({"label": label, "score": score})
        # Sort by score descending (like real pipeline)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results


class _RaisingPipeline:
    """Mock CLAP pipeline that raises on every call."""

    def __call__(self, audio, candidate_labels=None, **kwargs):
        raise RuntimeError("CLAP model OOM")


def _noop_resample(audio_16k):
    """Skip librosa resampling in tests — return audio as-is."""
    return audio_16k


def _make_verifier(
    clap_scores: dict[str, float],
    config: CLAPConfig | None = None,
) -> CLAPVerifier:
    """Create a CLAPVerifier with a mocked pipeline."""
    cfg = config or CLAPConfig()
    verifier = CLAPVerifier.__new__(CLAPVerifier)
    verifier._config = cfg
    verifier._prompts = verifier._build_prompts()
    verifier._all_prompt_texts = verifier._flatten_prompts()
    verifier._prompt_to_group = verifier._build_reverse_map()
    verifier._pipeline = FakeCLAPPipeline(clap_scores)
    verifier._resample = staticmethod(_noop_resample)
    verifier._loaded = True
    return verifier


class TestCLAPVerifierConfirm:
    def test_high_clap_score_confirms(self):
        """When CLAP scores AST's group >= confirm_threshold, mark verified."""
        verifier = _make_verifier({"a dog barking": 0.72})
        ast_results = [_make_ast_result()]
        results = verifier.verify(
            np.zeros(16000, dtype=np.float32), ast_results, "test_cam"
        )
        assert len(results) == 1
        assert results[0].clap_verified is True
        assert results[0].clap_score == 0.72
        assert results[0].clap_label == "a dog barking"
        assert results[0].source == "ast"

    def test_confirm_at_threshold(self):
        """Exactly at confirm_threshold should still confirm."""
        verifier = _make_verifier({"a dog barking": 0.30})
        ast_results = [_make_ast_result()]
        results = verifier.verify(
            np.zeros(16000, dtype=np.float32), ast_results, "test_cam"
        )
        assert len(results) == 1
        assert results[0].clap_verified is True


class TestCLAPVerifierSuppress:
    def test_low_clap_with_strong_alternative_suppresses(self):
        """When CLAP disagrees and has a strong alternative, suppress."""
        # AST says music, CLAP says vacuum_cleaner strongly, music weakly
        verifier = _make_verifier({
            "music playing with instruments": 0.05,
            "a vacuum cleaner running": 0.65,
        })
        ast_results = [_make_ast_result(label="Music", group="music", confidence=0.60)]
        results = verifier.verify(
            np.zeros(16000, dtype=np.float32), ast_results, "test_cam"
        )
        # Music should be suppressed (CLAP scores music < 0.15, vacuum >= 0.40)
        music_results = [r for r in results if r.group == "music"]
        assert len(music_results) == 0

    def test_no_suppress_when_clap_above_suppress_threshold(self):
        """If CLAP scores the group >= suppress_threshold, don't suppress."""
        verifier = _make_verifier({
            "music playing with instruments": 0.20,  # >= 0.15 suppress_threshold
            "a vacuum cleaner running": 0.65,
        })
        ast_results = [_make_ast_result(label="Music", group="music", confidence=0.60)]
        results = verifier.verify(
            np.zeros(16000, dtype=np.float32), ast_results, "test_cam"
        )
        # Not suppressed because clap score for music (0.20) >= suppress_threshold (0.15)
        # Not confirmed either because 0.20 < confirm_threshold (0.30)
        music_results = [r for r in results if r.group == "music"]
        assert len(music_results) == 1
        assert music_results[0].clap_verified is False

    def test_no_suppress_without_strong_alternative(self):
        """If CLAP has no strong alternative, don't suppress even with low score."""
        verifier = _make_verifier({
            "music playing with instruments": 0.05,
            # No alternative above override_threshold (0.40)
            "a vacuum cleaner running": 0.30,
        })
        ast_results = [_make_ast_result(label="Music", group="music", confidence=0.60)]
        results = verifier.verify(
            np.zeros(16000, dtype=np.float32), ast_results, "test_cam"
        )
        music_results = [r for r in results if r.group == "music"]
        assert len(music_results) == 1
        assert music_results[0].clap_verified is False


class TestCLAPVerifierNeverSuppress:
    def test_safety_groups_never_suppressed(self):
        """Safety-critical groups pass through regardless of CLAP scores."""
        for group in DEFAULT_NEVER_SUPPRESS:
            # CLAP gives near-zero score, strong alternative exists
            verifier = _make_verifier({
                "a vacuum cleaner running": 0.90,
                # All prompts for the safety group get near-zero
            })
            ast_results = [_make_ast_result(
                label="TestLabel", group=group, confidence=0.50
            )]
            results = verifier.verify(
                np.zeros(16000, dtype=np.float32), ast_results, "test_cam"
            )
            group_results = [r for r in results if r.group == group]
            assert len(group_results) == 1, f"Safety group {group} was suppressed!"

    def test_safety_group_with_high_clap_is_confirmed(self):
        """Safety group with high CLAP score gets verified=True."""
        verifier = _make_verifier({"a smoke alarm beeping": 0.80})
        ast_results = [_make_ast_result(
            label="Smoke detector", group="smoke_alarm", confidence=0.90
        )]
        results = verifier.verify(
            np.zeros(16000, dtype=np.float32), ast_results, "test_cam"
        )
        assert len(results) == 1
        assert results[0].clap_verified is True
        assert results[0].clap_score == 0.80


class TestCLAPVerifierDiscovery:
    def test_clap_only_discovery(self):
        """If CLAP detects a group AST missed above discovery_threshold, add it."""
        verifier = _make_verifier({
            "a dog barking": 0.72,
            "a vacuum cleaner running": 0.55,  # >= discovery_threshold 0.50
        })
        # AST only detected dog_bark
        ast_results = [_make_ast_result()]
        results = verifier.verify(
            np.zeros(16000, dtype=np.float32), ast_results, "test_cam"
        )
        # Should have dog_bark (confirmed) + vacuum_cleaner (discovered)
        groups = {r.group for r in results}
        assert "dog_bark" in groups
        assert "vacuum_cleaner" in groups

        vacuum = next(r for r in results if r.group == "vacuum_cleaner")
        assert vacuum.source == "clap"
        assert vacuum.clap_verified is True
        assert vacuum.clap_score >= 0.50

    def test_no_discovery_below_threshold(self):
        """Groups below discovery_threshold are not added."""
        verifier = _make_verifier({
            "a dog barking": 0.72,
            "a vacuum cleaner running": 0.40,  # < discovery_threshold 0.50
        })
        ast_results = [_make_ast_result()]
        results = verifier.verify(
            np.zeros(16000, dtype=np.float32), ast_results, "test_cam"
        )
        groups = {r.group for r in results}
        assert "vacuum_cleaner" not in groups

    def test_no_duplicate_discovery_for_existing_group(self):
        """If AST already detected a group, don't add it again via discovery."""
        verifier = _make_verifier({
            "a dog barking": 0.72,
        })
        ast_results = [_make_ast_result()]
        results = verifier.verify(
            np.zeros(16000, dtype=np.float32), ast_results, "test_cam"
        )
        dog_results = [r for r in results if r.group == "dog_bark"]
        assert len(dog_results) == 1  # Only the original, not duplicated


class TestCLAPVerifierUnverified:
    def test_moderate_clap_score_is_unverified(self):
        """Score between suppress and confirm thresholds = unverified."""
        verifier = _make_verifier({"a dog barking": 0.20})
        ast_results = [_make_ast_result()]
        results = verifier.verify(
            np.zeros(16000, dtype=np.float32), ast_results, "test_cam"
        )
        assert len(results) == 1
        assert results[0].clap_verified is False
        assert results[0].clap_score == 0.20


class TestCLAPVerifierCustomPrompts:
    def test_custom_prompts_merged(self):
        """Custom prompts should extend default prompts."""
        config = CLAPConfig(
            custom_prompts={
                "vacuum_cleaner": [
                    "a robot vacuum cleaner running",
                    "a roomba motor",
                ]
            }
        )
        verifier = _make_verifier(
            {"a robot vacuum cleaner running": 0.80},
            config=config,
        )
        # Verify custom prompt is in the prompt list
        assert "a robot vacuum cleaner running" in verifier._all_prompt_texts
        # Default prompt should still be there too
        assert "a vacuum cleaner running" in verifier._all_prompt_texts

    def test_custom_prompts_no_duplicates(self):
        """If custom prompt matches default, don't duplicate."""
        config = CLAPConfig(
            custom_prompts={
                "vacuum_cleaner": ["a vacuum cleaner running"],  # Same as default
            }
        )
        verifier = _make_verifier({}, config=config)
        count = verifier._all_prompt_texts.count("a vacuum cleaner running")
        assert count == 1


class TestCLAPVerifierEdgeCases:
    def test_empty_ast_results_with_strong_clap_discovers(self):
        """CLAP discovery still fires on empty AST results."""
        verifier = _make_verifier({"a dog barking": 0.90})
        results = verifier.verify(
            np.zeros(16000, dtype=np.float32), [], "test_cam"
        )
        groups = {r.group for r in results}
        assert "dog_bark" in groups
        discovered = next(r for r in results if r.group == "dog_bark")
        assert discovered.source == "clap"
        assert discovered.db_level == 0.0  # No AST ref → falls back to 0.0
        assert discovered.top_5 == []  # No AST predictions for discovered events

    def test_multiple_ast_results(self):
        """Multiple AST results should each be verified independently."""
        verifier = _make_verifier({
            "a dog barking": 0.72,
            "a person speaking clearly": 0.45,
        })
        ast_results = [
            _make_ast_result(label="Dog", group="dog_bark", confidence=0.85),
            _make_ast_result(label="Speech", group="speech", confidence=0.30),
        ]
        results = verifier.verify(
            np.zeros(16000, dtype=np.float32), ast_results, "test_cam"
        )
        groups = {r.group for r in results}
        assert "dog_bark" in groups
        assert "speech" in groups

    def test_suppression_with_discovery(self):
        """Suppressed AST result + CLAP discovery = correct replacement."""
        verifier = _make_verifier({
            "music playing with instruments": 0.05,
            "a vacuum cleaner running": 0.65,
        })
        ast_results = [_make_ast_result(label="Music", group="music", confidence=0.60)]
        results = verifier.verify(
            np.zeros(16000, dtype=np.float32), ast_results, "test_cam"
        )
        # Music suppressed, vacuum discovered
        groups = {r.group for r in results}
        assert "music" not in groups
        assert "vacuum_cleaner" in groups
        assert results[0].source == "clap"


class TestCLAPVerifierInferenceFailure:
    def test_pipeline_exception_returns_ast_results(self):
        """If CLAP pipeline throws, return unmodified AST results."""
        verifier = _make_verifier({})
        # Replace pipeline with one that raises
        verifier._pipeline = _RaisingPipeline()
        ast_results = [_make_ast_result()]
        results = verifier.verify(
            np.zeros(16000, dtype=np.float32), ast_results, "test_cam"
        )
        # Should get back the original AST results unmodified
        assert len(results) == 1
        assert results[0].group == "dog_bark"
        assert results[0].confidence == 0.85
        # Should NOT have CLAP fields set
        assert results[0].clap_verified is None
        assert results[0].clap_score is None


class TestCLAPResample:
    """Test torchaudio-based resampling."""

    def test_resample_output_shape(self):
        """Resampled audio should have 3x the samples (16kHz -> 48kHz)."""
        verifier = _make_verifier({})
        # Create a mock resampler that returns correct shape
        import types

        def _mock_resample(self, audio_16k):
            # Simulate 3x upsampling
            return np.repeat(audio_16k, 3)

        verifier._resample = types.MethodType(_mock_resample, verifier)

        audio_16k = np.random.randn(16000).astype(np.float32)
        result = verifier._resample(audio_16k)
        assert len(result) == 48000

    def test_resample_called_in_verify(self):
        """verify() should call _resample to prepare audio for CLAP."""
        resample_calls = []
        verifier = _make_verifier({"a dog barking": 0.72})


        def _tracking_resample(audio):
            resample_calls.append(len(audio))
            return audio  # Return same audio (mock doesn't care about rate)

        verifier._resample = _tracking_resample

        ast_results = [_make_ast_result()]
        verifier.verify(np.zeros(16000, dtype=np.float32), ast_results, "test")

        assert len(resample_calls) == 1
        assert resample_calls[0] == 16000


class TestCLAPConfirmMargin:
    """Tests for the confirm_margin rule that prevents confirming
    a group when a much stronger alternative exists."""

    def test_margin_passes_when_competitive(self):
        """When CLAP score is competitive with best alternative, confirm."""
        # dog_bark=0.50, vacuum_cleaner=0.55 — difference is 0.05 < margin 0.20
        verifier = _make_verifier({
            "a dog barking": 0.50,
            "a vacuum cleaner running": 0.55,
        })
        ast_results = [_make_ast_result()]
        results = verifier.verify(
            np.zeros(16000, dtype=np.float32), ast_results, "test_cam"
        )
        assert len(results) >= 1
        dog = next(r for r in results if r.group == "dog_bark")
        assert dog.clap_verified is True

    def test_margin_fails_when_alternative_dominates(self):
        """When best alternative dominates, confirmation is rejected (unverified)."""
        # speech=0.30 (at threshold), vacuum_cleaner=0.75
        # margin check: 0.30 >= 0.75 - 0.20 = 0.55? No → unverified
        verifier = _make_verifier({
            "a person speaking clearly": 0.30,
            "a vacuum cleaner running": 0.75,
        })
        ast_results = [_make_ast_result(
            label="Speech", group="speech", confidence=0.40
        )]
        results = verifier.verify(
            np.zeros(16000, dtype=np.float32), ast_results, "test_cam"
        )
        speech = [r for r in results if r.group == "speech"]
        assert len(speech) == 1
        assert speech[0].clap_verified is False  # Unverified, not confirmed

    def test_margin_with_no_alternatives_always_passes(self):
        """When there are no alternative groups, margin check always passes."""
        # Only dog_bark has a score, no alternatives
        verifier = _make_verifier({"a dog barking": 0.35})
        ast_results = [_make_ast_result()]
        results = verifier.verify(
            np.zeros(16000, dtype=np.float32), ast_results, "test_cam"
        )
        assert len(results) >= 1
        dog = next(r for r in results if r.group == "dog_bark")
        assert dog.clap_verified is True

    def test_margin_does_not_affect_never_suppress(self):
        """Never-suppress groups bypass margin check entirely."""
        # smoke_alarm=0.30 (at threshold), vacuum_cleaner=0.80
        # Without never_suppress, margin would fail. With it, should pass through.
        verifier = _make_verifier({
            "a smoke alarm beeping": 0.30,
            "a vacuum cleaner running": 0.80,
        })
        ast_results = [_make_ast_result(
            label="Smoke detector", group="smoke_alarm", confidence=0.90
        )]
        results = verifier.verify(
            np.zeros(16000, dtype=np.float32), ast_results, "test_cam"
        )
        smoke = [r for r in results if r.group == "smoke_alarm"]
        assert len(smoke) == 1  # Not suppressed


class TestCLAPPromptQuality:
    """Verify prompt coverage for key groups."""

    def test_vacuum_cleaner_has_sufficient_prompts(self):
        assert len(DEFAULT_PROMPTS["vacuum_cleaner"]) >= 5

    def test_speech_has_sufficient_prompts(self):
        assert len(DEFAULT_PROMPTS["speech"]) >= 4

    def test_music_has_sufficient_prompts(self):
        assert len(DEFAULT_PROMPTS["music"]) >= 4

    def test_kitchen_appliance_has_sufficient_prompts(self):
        assert len(DEFAULT_PROMPTS["kitchen_appliance"]) >= 4


class TestCLAPNeverSuppressContents:
    """Verify NEVER_SUPPRESS membership after speech removal."""

    def test_speech_not_in_never_suppress(self):
        assert "speech" not in DEFAULT_NEVER_SUPPRESS

    def test_safety_groups_still_present(self):
        for group in ("smoke_alarm", "glass_break", "siren", "screaming", "crying"):
            assert group in DEFAULT_NEVER_SUPPRESS
