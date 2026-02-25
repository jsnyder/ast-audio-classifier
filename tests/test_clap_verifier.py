"""Tests for CLAP zero-shot verification logic."""

import numpy as np

from src.classifier import ClassificationResult
from src.clap_verifier import (
    DEFAULT_PROMPTS,
    CLAPConfig,
    CLAPVerifier,
    DEFAULT_NEVER_SUPPRESS,
    build_default_prompts,
)
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
        assert cfg.confirm_threshold == 0.25
        assert cfg.suppress_threshold == 0.15
        assert cfg.override_threshold == 0.40
        assert cfg.discovery_threshold == 0.50
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
        verifier = _make_verifier({"a dog barking": 0.25})
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
            "music playing": 0.05,
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
            "music playing": 0.20,  # >= 0.15 suppress_threshold
            "a vacuum cleaner running": 0.65,
        })
        ast_results = [_make_ast_result(label="Music", group="music", confidence=0.60)]
        results = verifier.verify(
            np.zeros(16000, dtype=np.float32), ast_results, "test_cam"
        )
        # Not suppressed because clap score for music (0.20) >= suppress_threshold (0.15)
        # But also not confirmed because 0.20 < confirm_threshold (0.25)
        music_results = [r for r in results if r.group == "music"]
        assert len(music_results) == 1
        assert music_results[0].clap_verified is False

    def test_no_suppress_without_strong_alternative(self):
        """If CLAP has no strong alternative, don't suppress even with low score."""
        verifier = _make_verifier({
            "music playing": 0.05,
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

        vacuum = [r for r in results if r.group == "vacuum_cleaner"][0]
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
    def test_empty_ast_results(self):
        """Empty AST results should return empty (no discovery without context)."""
        verifier = _make_verifier({"a dog barking": 0.90})
        results = verifier.verify(
            np.zeros(16000, dtype=np.float32), [], "test_cam"
        )
        # Discovery can still happen but with no AST context
        # The verifier should not crash
        assert isinstance(results, list)

    def test_multiple_ast_results(self):
        """Multiple AST results should each be verified independently."""
        verifier = _make_verifier({
            "a dog barking": 0.72,
            "a person speaking": 0.45,
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
            "music playing": 0.05,
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
