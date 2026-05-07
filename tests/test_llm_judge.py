"""Tests for LLM Judge audio evaluation."""

import json
import os
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.classifier import ClassificationResult
from src.config import LLMJudgeConfig
from src.llm_judge import LLMJudge


@pytest.fixture()
def judge_config(tmp_path):
    """LLMJudgeConfig pointing at a temp directory."""
    return LLMJudgeConfig(
        enabled=True,
        api_base="https://litellm.example.com/v1",
        api_key="sk-test",
        model="gemini-2.5-flash",
        sample_rate=1.0,  # Always sample in tests
        clip_dir=str(tmp_path / "clips"),
        max_clips=5,
        timeout_seconds=10,
    )


@pytest.fixture()
def sample_results():
    """Sample ClassificationResult list."""
    return [
        ClassificationResult(
            label="Dog",
            group="dog_bark",
            confidence=0.85,
            top_5=[("Dog", 0.85), ("Bark", 0.60), ("Cat", 0.05), ("Speech", 0.03), ("Music", 0.01)],
            db_level=-25.3,
            clap_verified=True,
            clap_score=0.72,
            clap_label="a dog barking",
        ),
    ]


@pytest.fixture()
def sample_audio():
    """3 seconds of 16kHz mono float32 audio (sine wave)."""
    t = np.linspace(0, 3, 48000, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


class TestShouldSample:
    def test_rate_1_always_true(self, judge_config):
        judge = LLMJudge(judge_config)
        assert all(judge.should_sample() for _ in range(100))

    def test_rate_0_always_false(self, judge_config):
        judge_config.sample_rate = 0.0
        judge = LLMJudge(judge_config)
        assert not any(judge.should_sample() for _ in range(100))

    def test_statistical_rate(self, judge_config):
        judge_config.sample_rate = 0.5
        judge = LLMJudge(judge_config)
        samples = [judge.should_sample() for _ in range(1000)]
        ratio = sum(samples) / len(samples)
        assert 0.40 < ratio < 0.60


class TestSaveWav:
    def test_creates_wav_file(self, judge_config, sample_audio):
        judge = LLMJudge(judge_config)
        path, wav_bytes = judge._save_wav(sample_audio, "living_room", "dog_bark")
        assert os.path.exists(path)
        assert path.endswith(".wav")
        assert len(wav_bytes) > 0

    def test_wav_format_correct(self, judge_config, sample_audio):
        judge = LLMJudge(judge_config)
        path, _wav_bytes = judge._save_wav(sample_audio, "front_porch", "glass_break")
        with wave.open(path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2  # 16-bit
            assert wf.getframerate() == 16000

    def test_filename_format(self, judge_config, sample_audio):
        judge = LLMJudge(judge_config)
        path, _wav_bytes = judge._save_wav(sample_audio, "back_yard", "cat_meow")
        filename = os.path.basename(path)
        assert filename.startswith("back_yard_")
        assert "cat_meow" in filename
        assert filename.endswith(".wav")

    def test_clip_dir_created_on_save(self, judge_config, sample_audio):
        """Directory is created lazily on first save, not at construction."""
        assert not os.path.exists(judge_config.clip_dir)
        judge = LLMJudge(judge_config)
        # Constructor should NOT create the directory
        assert not os.path.exists(judge_config.clip_dir)
        # First save creates it
        judge._save_wav(sample_audio, "cam", "test")
        assert os.path.isdir(judge_config.clip_dir)

    def test_sanitizes_camera_name(self, judge_config, sample_audio):
        """Camera names with special chars are sanitized in filenames."""
        judge = LLMJudge(judge_config)
        path, _wav_bytes = judge._save_wav(sample_audio, "../../etc/evil", "dog_bark")
        filename = os.path.basename(path)
        # Path traversal chars should be replaced with underscores
        assert ".." not in filename
        assert "/" not in filename

    def test_wav_bytes_match_file(self, judge_config, sample_audio):
        """The returned bytes should match what was written to disk."""
        judge = LLMJudge(judge_config)
        path, wav_bytes = judge._save_wav(sample_audio, "cam", "test")
        with open(path, "rb") as f:
            assert f.read() == wav_bytes


class TestBuildPrompt:
    def test_includes_context(self, judge_config, sample_results):
        judge = LLMJudge(judge_config)
        prompt = judge._build_prompt(sample_results, "living_room")
        assert "living_room" in prompt
        assert "dog_bark" in prompt
        assert "0.85" in prompt
        assert "True" in prompt or "true" in prompt.lower()
        assert "0.72" in prompt

    def test_multiple_results(self, judge_config):
        judge = LLMJudge(judge_config)
        results = [
            ClassificationResult(
                label="Dog",
                group="dog_bark",
                confidence=0.85,
                top_5=[],
                db_level=-25.0,
                clap_verified=True,
                clap_score=0.72,
            ),
            ClassificationResult(
                label="Cat",
                group="cat_meow",
                confidence=0.45,
                top_5=[],
                db_level=-25.0,
                clap_verified=False,
                clap_score=0.18,
            ),
        ]
        prompt = judge._build_prompt(results, "backyard")
        assert "dog_bark" in prompt
        assert "cat_meow" in prompt


class TestStripMarkdown:
    def test_plain_json_unchanged(self, judge_config):
        judge = LLMJudge(judge_config)
        text = '{"verdicts": [{"verdict": "correct"}]}'
        assert judge._strip_markdown(text) == text

    def test_strips_json_fence(self, judge_config):
        judge = LLMJudge(judge_config)
        text = '```json\n{"verdicts": [{"verdict": "correct"}]}\n```'
        assert json.loads(judge._strip_markdown(text))["verdicts"][0]["verdict"] == "correct"

    def test_strips_plain_fence(self, judge_config):
        judge = LLMJudge(judge_config)
        text = '```\n{"verdicts": [{"verdict": "correct"}]}\n```'
        assert json.loads(judge._strip_markdown(text))["verdicts"][0]["verdict"] == "correct"

    def test_handles_whitespace(self, judge_config):
        judge = LLMJudge(judge_config)
        text = '  ```json\n{"key": "value"}\n```  '
        result = judge._strip_markdown(text)
        assert json.loads(result) == {"key": "value"}

    def test_parse_response_with_markdown_fences(self, judge_config):
        """Integration: _parse_response should handle markdown-wrapped JSON."""
        judge = LLMJudge(judge_config)
        response = '```json\n{"verdicts": [{"group": "dog_bark", "verdict": "correct", "confidence": 0.9}]}\n```'
        verdicts = judge._parse_response(response)
        assert len(verdicts) == 1
        assert verdicts[0]["verdict"] == "correct"


class TestParseResponse:
    def test_valid_json(self, judge_config):
        judge = LLMJudge(judge_config)
        response_text = json.dumps(
            {
                "verdicts": [
                    {
                        "group": "dog_bark",
                        "verdict": "correct",
                        "actual_sound": "a dog barking loudly",
                        "confidence": 0.95,
                        "notes": "Clear bark audio",
                    }
                ]
            }
        )
        verdicts = judge._parse_response(response_text)
        assert len(verdicts) == 1
        assert verdicts[0]["verdict"] == "correct"
        assert verdicts[0]["actual_sound"] == "a dog barking loudly"
        assert verdicts[0]["confidence"] == 0.95

    def test_malformed_json(self, judge_config):
        judge = LLMJudge(judge_config)
        verdicts = judge._parse_response("not valid json at all")
        assert len(verdicts) == 1
        assert verdicts[0]["verdict"] == "error"
        assert "parse" in verdicts[0]["notes"].lower() or "failed" in verdicts[0]["notes"].lower()

    def test_missing_fields(self, judge_config):
        judge = LLMJudge(judge_config)
        response_text = json.dumps({"verdicts": [{"verdict": "correct"}]})
        verdicts = judge._parse_response(response_text)
        assert len(verdicts) == 1
        assert verdicts[0]["verdict"] == "correct"
        # Missing fields should get defaults
        assert "actual_sound" in verdicts[0]
        assert "confidence" in verdicts[0]

    def test_single_verdict_not_in_list(self, judge_config):
        """Handle LLM returning a single verdict object instead of a list."""
        judge = LLMJudge(judge_config)
        response_text = json.dumps(
            {
                "verdict": "correct",
                "actual_sound": "dog bark",
                "confidence": 0.9,
                "notes": "clear bark",
            }
        )
        verdicts = judge._parse_response(response_text)
        assert len(verdicts) == 1
        assert verdicts[0]["verdict"] == "correct"

    def test_confidence_as_string_coerced_to_zero(self, judge_config):
        """LLM may return confidence as a string — should default to 0.0."""
        judge = LLMJudge(judge_config)
        response_text = json.dumps({"verdicts": [{"verdict": "correct", "confidence": "high"}]})
        verdicts = judge._parse_response(response_text)
        assert verdicts[0]["confidence"] == 0.0
        assert isinstance(verdicts[0]["confidence"], float)

    def test_verdicts_matched_by_group_name(self, judge_config):
        """Verdicts should be matchable by group name, not position."""
        judge = LLMJudge(judge_config)
        response_text = json.dumps(
            {
                "verdicts": [
                    {
                        "group": "cat_meow",
                        "verdict": "incorrect",
                        "actual_sound": "tv",
                        "confidence": 0.8,
                        "notes": "tv audio",
                    },
                    {
                        "group": "dog_bark",
                        "verdict": "correct",
                        "actual_sound": "bark",
                        "confidence": 0.9,
                        "notes": "clear",
                    },
                ]
            }
        )
        verdicts = judge._parse_response(response_text)
        assert len(verdicts) == 2
        assert verdicts[0]["group"] == "cat_meow"
        assert verdicts[1]["group"] == "dog_bark"


class TestPruneClips:
    def test_removes_oldest(self, judge_config, sample_audio):
        judge = LLMJudge(judge_config)
        # Create max_clips + 2 files
        for i in range(judge_config.max_clips + 2):
            judge._save_wav(sample_audio, "cam", f"group_{i}")

        judge._prune_clips()
        remaining = os.listdir(judge_config.clip_dir)
        assert len(remaining) <= judge_config.max_clips

    def test_noop_under_limit(self, judge_config, sample_audio):
        judge = LLMJudge(judge_config)
        # Create fewer than max_clips files
        for i in range(3):
            judge._save_wav(sample_audio, "cam", f"group_{i}")

        judge._prune_clips()
        remaining = os.listdir(judge_config.clip_dir)
        assert len(remaining) == 3


class TestShouldPrune:
    def test_triggers_at_interval(self, judge_config):
        judge = LLMJudge(judge_config)
        # Default interval is 100
        triggered = []
        for _ in range(250):
            triggered.append(judge._should_prune())
        # Should trigger at eval 100, 200 → exactly 2 times
        assert sum(triggered) == 2

    def test_first_99_false(self, judge_config):
        judge = LLMJudge(judge_config)
        results = [judge._should_prune() for _ in range(99)]
        assert not any(results)

    def test_100th_true(self, judge_config):
        judge = LLMJudge(judge_config)
        for _ in range(99):
            judge._should_prune()
        assert judge._should_prune() is True


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_logs_to_openobserve(self, judge_config, sample_audio, sample_results):
        judge = LLMJudge(judge_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "verdicts": [
                    {
                        "group": "dog_bark",
                        "verdict": "correct",
                        "actual_sound": "a dog barking",
                        "confidence": 0.9,
                        "notes": "Clear bark",
                    }
                ]
            }
        )

        with (
            patch.object(
                judge._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            patch("src.llm_judge.log_event") as mock_log,
        ):
            await judge.evaluate(sample_audio, sample_results, "living_room")
            mock_log.assert_called()
            call_kwargs = mock_log.call_args
            assert call_kwargs[0][0] == "llm_judge"  # event_type
            assert call_kwargs[1]["camera"] == "living_room"
            assert call_kwargs[1]["llm_verdict"] == "correct"

    @pytest.mark.asyncio
    async def test_handles_api_error(self, judge_config, sample_audio, sample_results):
        judge = LLMJudge(judge_config)

        with patch.object(
            judge._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=Exception("API timeout"),
        ):
            # Should not raise — fire-and-forget semantics
            await judge.evaluate(sample_audio, sample_results, "living_room")

    @pytest.mark.asyncio
    async def test_saves_wav_clip(self, judge_config, sample_audio, sample_results):
        judge = LLMJudge(judge_config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "verdicts": [
                    {
                        "verdict": "correct",
                        "actual_sound": "bark",
                        "confidence": 0.9,
                        "notes": "ok",
                    }
                ]
            }
        )

        with (
            patch.object(
                judge._client.chat.completions,
                "create",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            patch("src.llm_judge.log_event"),
        ):
            await judge.evaluate(sample_audio, sample_results, "test_cam")

        all_files = os.listdir(judge_config.clip_dir)
        wav_files = [f for f in all_files if f.endswith(".wav")]
        json_files = [f for f in all_files if f.endswith(".json")]
        assert len(wav_files) == 1
        assert len(json_files) == 1
        assert wav_files[0].rsplit(".", 1)[0] == json_files[0].rsplit(".", 1)[0]


class TestLLMJudgeConfigValidation:
    def test_sample_rate_out_of_range(self, tmp_path):
        with pytest.raises(ValueError, match="sample_rate"):
            LLMJudgeConfig(sample_rate=1.5, api_base="https://example.com/v1")

    def test_max_clips_zero(self, tmp_path):
        with pytest.raises(ValueError, match="max_clips"):
            LLMJudgeConfig(max_clips=0, api_base="https://example.com/v1")

    def test_timeout_zero(self, tmp_path):
        with pytest.raises(ValueError, match="timeout_seconds"):
            LLMJudgeConfig(timeout_seconds=0, api_base="https://example.com/v1")

    def test_enabled_without_api_base(self):
        with pytest.raises(ValueError, match="api_base"):
            LLMJudgeConfig(enabled=True, api_base="")

    def test_disabled_without_api_base_ok(self):
        cfg = LLMJudgeConfig(enabled=False, api_base="")
        assert cfg.api_base == ""
