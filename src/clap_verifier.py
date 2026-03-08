"""CLAP zero-shot verification for AST classification results.

Uses CLAP (Contrastive Language-Audio Pretraining) to verify, suppress,
or discover audio events that AST may misclassify or miss entirely.

Three outcomes per AST result:
  - Confirmed: CLAP agrees with AST's group classification
  - Suppressed: CLAP strongly disagrees and scores an alternative higher
  - Unverified: Neither condition met, passes through with flag

Safety-critical groups (smoke_alarm, glass_break, etc.) are never suppressed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np

from .classifier import ClassificationResult
from .labels import LABEL_GROUPS

logger = logging.getLogger(__name__)

# CLAP expects 48kHz audio
CLAP_SAMPLE_RATE = 48000
AST_SAMPLE_RATE = 16000

# Default never-suppress groups (safety-critical)
DEFAULT_NEVER_SUPPRESS = frozenset({
    "smoke_alarm",
    "glass_break",
    "siren",
    "screaming",
    "crying",
})


def _group_to_prompts(group: str) -> list[str]:
    """Convert a group name to natural language prompts.

    Uses the AudioSet labels in the group as the basis for prompt generation.
    """
    # Map group names to human-readable descriptions
    prompt_templates = {
        "smoke_alarm": ["a smoke alarm beeping", "a fire alarm going off"],
        "glass_break": ["glass shattering", "glass breaking"],
        "siren": ["a siren wailing", "an emergency vehicle siren"],
        "gunshot_explosion": ["a gunshot", "an explosion"],
        "screaming": ["a person screaming", "someone shouting"],
        "dog_bark": ["a dog barking"],
        "cat_meow": ["a cat meowing"],
        "crying": ["a baby crying", "someone crying"],
        "speech": ["a person speaking clearly", "people talking in a conversation", "a human voice speaking words", "someone talking in a room"],
        "cough_sneeze": ["a person coughing", "someone sneezing"],
        "footsteps": ["footsteps walking"],
        "doorbell": ["a doorbell ringing"],
        "knock": ["someone knocking on a door"],
        "door": ["a door opening or closing", "a door slamming"],
        "cabinet": ["a cupboard or drawer opening"],
        "rain_storm": ["rain falling", "a thunderstorm"],
        "music": ["music playing with instruments", "a song with singing and melody", "rhythmic music with drums and bass", "background music from a speaker"],
        "vehicle": ["a vehicle driving", "a car passing by"],
        "car_horn": ["a car horn honking"],
        "aircraft": ["an airplane flying overhead", "a jet engine"],
        "vacuum_cleaner": ["a vacuum cleaner running", "an autonomous vacuum cleaner operating", "a robot vacuum motor whine", "a constant mechanical humming from a vacuum", "a robotic vacuum rolling on the floor", "a continuous droning motor hum", "a roborock robot vacuum cleaning"],
        "water_running": ["water running from a faucet", "water splashing"],
        "kitchen_appliance": ["dishes clanking in a kitchen", "plates and silverware clattering", "a blender running in a kitchen", "a microwave beeping", "cooking sounds with pots and pans"],
        "power_tools": ["a power tool running", "a drill or saw"],
        "alarm_beep": ["an alarm beeping", "a buzzer sounding"],
        "hvac_mechanical": ["an air conditioning unit", "a mechanical fan running", "a furnace blower running", "a heating system running", "white noise from a fan"],
        "mechanical_anomaly": ["a mechanical rattling noise", "equipment vibrating"],
        "water_leak": ["water dripping", "water trickling"],
        "electrical_anomaly": ["an electrical buzzing noise", "a humming sound"],
        "media": [
            "television audio playing",
            "a TV show or movie soundtrack",
            "audio from a television speaker",
        ],
    }
    if group in prompt_templates:
        return prompt_templates[group]
    # Fallback: convert group name to natural language
    readable = group.replace("_", " ")
    return [f"the sound of {readable}"]


def build_default_prompts() -> dict[str, list[str]]:
    """Build default CLAP prompts from all LABEL_GROUPS keys."""
    return {group: _group_to_prompts(group) for group in LABEL_GROUPS}


# Module-level default prompts
DEFAULT_PROMPTS: dict[str, list[str]] = build_default_prompts()


@dataclass
class CLAPConfig:
    """Configuration for CLAP verification."""

    enabled: bool = True
    model: str = "laion/clap-htsat-fused"
    confirm_threshold: float = 0.30
    suppress_threshold: float = 0.15
    override_threshold: float = 0.40
    discovery_threshold: float = 0.50
    confirm_margin: float = 0.20
    ast_bypass_threshold: float = 0.80  # Skip CLAP suppression when AST confidence exceeds this
    never_suppress: frozenset[str] = DEFAULT_NEVER_SUPPRESS
    custom_prompts: dict[str, list[str]] | None = None


class CLAPVerifier:
    """Verifies AST classification results using CLAP zero-shot inference.

    Loads the CLAP model once at init and reuses for all verifications.
    Thread-safe when called via asyncio.to_thread with a semaphore.
    """

    def __init__(self, config: CLAPConfig | None = None) -> None:
        self._loaded = False
        self._config = config or CLAPConfig()
        self._prompts = self._build_prompts()
        self._all_prompt_texts = self._flatten_prompts()
        self._prompt_to_group = self._build_reverse_map()
        self._last_suppressed: list[ClassificationResult] = []

        from transformers import pipeline as hf_pipeline

        logger.info("Loading CLAP model: %s", self._config.model)
        self._pipeline = hf_pipeline(
            "zero-shot-audio-classification",
            model=self._config.model,
            device=-1,  # CPU
        )
        import torchaudio
        self._resampler = torchaudio.transforms.Resample(AST_SAMPLE_RATE, CLAP_SAMPLE_RATE)
        self._loaded = True
        logger.info("CLAP model loaded successfully")

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def last_suppressed(self) -> list[ClassificationResult]:
        """Results suppressed by the most recent verify() call."""
        return self._last_suppressed

    def _build_prompts(self) -> dict[str, list[str]]:
        """Merge default prompts with custom prompts from config."""
        prompts = dict(DEFAULT_PROMPTS)
        if self._config.custom_prompts:
            for group, texts in self._config.custom_prompts.items():
                if group in prompts:
                    # Extend existing prompts with custom ones (no duplicates)
                    existing = set(prompts[group])
                    prompts[group] = prompts[group] + [
                        t for t in texts if t not in existing
                    ]
                else:
                    prompts[group] = texts
        return prompts

    def _flatten_prompts(self) -> list[str]:
        """Get flat list of all prompt texts for CLAP inference."""
        texts = []
        seen = set()
        for group_texts in self._prompts.values():
            for t in group_texts:
                if t not in seen:
                    texts.append(t)
                    seen.add(t)
        return texts

    def _build_reverse_map(self) -> dict[str, str]:
        """Map prompt text -> group name for score aggregation."""
        mapping = {}
        for group, texts in self._prompts.items():
            for t in texts:
                mapping[t] = group
        return mapping

    def _score_by_group(
        self, clap_results: list[dict],
    ) -> dict[str, float]:
        """Aggregate CLAP scores by group, taking max per group."""
        group_scores: dict[str, float] = {}
        for item in clap_results:
            label: str = item["label"]
            score: float = float(item["score"])
            group = self._prompt_to_group.get(label)
            if group is not None and (group not in group_scores or score > group_scores[group]):
                group_scores[group] = score
        return group_scores

    def _best_clap_label(
        self, group: str, clap_results: list[dict],
    ) -> tuple[str, float]:
        """Get the best scoring CLAP label for a given group."""
        best_label = ""
        best_score = 0.0
        group_prompts = set(self._prompts.get(group, []))
        for item in clap_results:
            score = float(item["score"])
            if item["label"] in group_prompts and score > best_score:
                best_label = item["label"]
                best_score = score
        return best_label, best_score

    def _resample(self, audio_16k: np.ndarray) -> np.ndarray:
        """Resample 16kHz audio to 48kHz for CLAP using torchaudio."""
        import torch
        tensor = torch.from_numpy(audio_16k)
        resampled = self._resampler(tensor)
        return resampled.numpy()

    def verify(
        self,
        audio_16k: np.ndarray,
        ast_results: list[ClassificationResult],
        camera_name: str,
    ) -> list[ClassificationResult]:
        """Verify AST results using CLAP zero-shot classification.

        Args:
            audio_16k: Float32 numpy array at 16kHz (same as AST input).
            ast_results: Classification results from AST.
            camera_name: Camera name for logging.

        Returns:
            List of verified/enriched ClassificationResult objects.
            Suppressed results are excluded from the list.
        """
        self._last_suppressed = []
        audio_48k = self._resample(audio_16k)

        # Run CLAP zero-shot against all prompts
        try:
            clap_results = self._pipeline(
                audio_48k,
                candidate_labels=self._all_prompt_texts,
            )
        except Exception:
            logger.exception(
                "[%s] CLAP inference failed, returning unmodified AST results",
                camera_name,
            )
            return ast_results

        group_scores = self._score_by_group(clap_results)

        verified_results: list[ClassificationResult] = []
        ast_groups = {r.group for r in ast_results}

        for result in ast_results:
            group = result.group
            clap_score = group_scores.get(group, 0.0)
            clap_label, _ = self._best_clap_label(group, clap_results)

            # Safety override: never suppress critical groups
            if group in self._config.never_suppress:
                verified_results.append(
                    replace(
                        result,
                        clap_verified=True if clap_score >= self._config.confirm_threshold else None,
                        clap_score=round(clap_score, 4) if clap_score > 0 else None,
                        clap_label=clap_label or None,
                    )
                )
                continue

            # Find best alternative score (used by both margin check and suppression)
            best_alt_group = None
            best_alt_score = 0.0
            for alt_group, alt_score in group_scores.items():
                if alt_group != group and alt_score > best_alt_score:
                    best_alt_group = alt_group
                    best_alt_score = alt_score

            # Check for confirmation (with margin gate)
            if clap_score >= self._config.confirm_threshold:
                # Margin check: CLAP score must be competitive with best alternative
                if clap_score >= best_alt_score - self._config.confirm_margin:
                    logger.info(
                        "[%s] CLAP confirmed %s (clap=%.3f, ast=%.3f)",
                        camera_name, group, clap_score, result.confidence,
                    )
                    verified_results.append(
                        replace(
                            result,
                            clap_verified=True,
                            clap_score=round(clap_score, 4),
                            clap_label=clap_label,
                        )
                    )
                    continue
                else:
                    logger.info(
                        "[%s] CLAP margin rejected %s (clap=%.3f, alt=%s=%.3f, margin=%.2f)",
                        camera_name, group, clap_score,
                        best_alt_group, best_alt_score, self._config.confirm_margin,
                    )
                    # Fall through to unverified path

            # AST confidence bypass: trust high-confidence AST over CLAP suppression
            # Only applies to safety-critical groups (never_suppress) — for non-safety
            # groups like speech/car_horn, trust CLAP's suppression even at high AST.
            if (
                result.confidence >= self._config.ast_bypass_threshold
                and clap_score < self._config.suppress_threshold
                and best_alt_score >= self._config.override_threshold
                and group in self._config.never_suppress
            ):
                logger.info(
                    "[%s] AST bypass: %s kept despite CLAP suppression "
                    "(ast=%.3f >= %.2f, clap=%.3f, alt=%s=%.3f)",
                    camera_name, group, result.confidence,
                    self._config.ast_bypass_threshold,
                    clap_score, best_alt_group, best_alt_score,
                )
                verified_results.append(
                    replace(
                        result,
                        clap_verified=False,
                        clap_score=round(clap_score, 4) if clap_score > 0 else None,
                        clap_label=clap_label or None,
                    )
                )
                continue

            if (
                clap_score < self._config.suppress_threshold
                and best_alt_score >= self._config.override_threshold
            ):
                logger.info(
                    "[%s] CLAP suppressed %s (clap=%.3f, alt=%s=%.3f, ast=%.3f)",
                    camera_name, group, clap_score,
                    best_alt_group, best_alt_score, result.confidence,
                )
                self._last_suppressed.append(
                    replace(
                        result,
                        clap_verified=False,
                        clap_score=round(clap_score, 4) if clap_score > 0 else None,
                        clap_label=clap_label or None,
                    )
                )
                continue  # Drop from verified results

            # Unverified: passes through with flag
            logger.debug(
                "[%s] CLAP unverified %s (clap=%.3f, ast=%.3f)",
                camera_name, group, clap_score, result.confidence,
            )
            verified_results.append(
                replace(
                    result,
                    clap_verified=False,
                    clap_score=round(clap_score, 4) if clap_score > 0 else None,
                    clap_label=clap_label or None,
                )
            )

        # CLAP-only discovery: groups CLAP detects that AST missed
        for group, score in group_scores.items():
            if group not in ast_groups and score >= self._config.discovery_threshold:
                clap_label, _ = self._best_clap_label(group, clap_results)
                # db_level comes from the audio frame, not the classifier
                ref = ast_results[0] if ast_results else None
                db_level = ref.db_level if ref else 0.0

                logger.info(
                    "[%s] CLAP discovered %s (clap=%.3f, label=%s)",
                    camera_name, group, score, clap_label,
                )
                verified_results.append(
                    ClassificationResult(
                        label=clap_label,
                        group=group,
                        confidence=round(score, 4),
                        top_5=[],
                        db_level=db_level,
                        clap_verified=True,
                        clap_score=round(score, 4),
                        clap_label=clap_label,
                        source="clap",
                    )
                )

        return verified_results
