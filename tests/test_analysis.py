"""
Tests for src/analysis.py (inference functions) using mocked model objects.

Real model integration tests are kept separate and are not run here.
Mocks demonstrate that the inference wrappers behave correctly; they
do NOT prove pretrained model inference works.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from src.analysis import CLIP_MAX_TOKENS, run_blip_caption, run_clip


def _dummy_image() -> Image.Image:
    return Image.new("RGB", (32, 32), color=(128, 128, 128))


# ── run_clip ──────────────────────────────────────────────────────────

class TestRunClip:
    def _mock_processor_and_model(self, token_count: int = 5, score: float = 0.25):
        processor = MagicMock()
        processor.tokenizer.return_value = {"input_ids": list(range(token_count))}
        processor.return_value = {
            "input_ids": torch.zeros(1, token_count, dtype=torch.long),
        }

        embed = torch.randn(1, 512)
        embed = embed / embed.norm(dim=-1, keepdim=True)

        model = MagicMock()
        outputs = MagicMock()
        # Set up so cosine_similarity returns ~score.
        # We use identical vectors for simplicity (cos_sim=1.0).
        outputs.image_embeds = embed
        outputs.text_embeds = embed
        model.return_value = outputs
        return processor, model

    def test_returns_float_on_success(self):
        proc, model = self._mock_processor_and_model()
        score, err = run_clip(_dummy_image(), "A dog on grass.", proc, model)
        assert err is None
        assert isinstance(score, float)

    def test_score_is_finite(self):
        proc, model = self._mock_processor_and_model()
        score, err = run_clip(_dummy_image(), "A dog on grass.", proc, model)
        # Allow tiny FP precision overshoot (e.g. 1.0000001 from float32).
        assert score is not None and -1.0 - 1e-5 <= score <= 1.0 + 1e-5

    def test_token_limit_exceeded_raises_value_error(self):
        proc, model = self._mock_processor_and_model(token_count=CLIP_MAX_TOKENS + 1)
        with pytest.raises(ValueError, match="too long"):
            run_clip(_dummy_image(), "x" * 200, proc, model)

    def test_token_limit_at_max_is_ok(self):
        proc, model = self._mock_processor_and_model(token_count=CLIP_MAX_TOKENS)
        score, err = run_clip(_dummy_image(), "valid claim", proc, model)
        assert err is None

    def test_inference_error_returns_none_and_message(self):
        proc = MagicMock()
        proc.tokenizer.return_value = {"input_ids": [1, 2, 3]}
        proc.side_effect = RuntimeError("GPU OOM")
        model = MagicMock()
        score, err = run_clip(_dummy_image(), "claim", proc, model)
        assert score is None
        assert err is not None
        assert "CLIP inference error" in err

    def test_model_forward_error_returns_none_and_message(self):
        proc = MagicMock()
        proc.tokenizer.return_value = {"input_ids": [1, 2, 3]}
        proc.return_value = {"input_ids": torch.zeros(1, 3, dtype=torch.long)}
        model = MagicMock()
        model.side_effect = RuntimeError("forward error")
        score, err = run_clip(_dummy_image(), "claim", proc, model)
        assert score is None
        assert err is not None


# ── run_blip_caption ──────────────────────────────────────────────────

class TestRunBlipCaption:
    def _mock_blip(self, caption: str = "a dog on grass"):
        processor = MagicMock()
        processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
        processor.decode.return_value = caption

        model = MagicMock()
        model.generate.return_value = torch.zeros(1, 10, dtype=torch.long)
        return processor, model

    def test_returns_string_on_success(self):
        proc, model = self._mock_blip()
        caption, err = run_blip_caption(_dummy_image(), proc, model)
        assert err is None
        assert isinstance(caption, str)
        assert len(caption) > 0

    def test_decode_called(self):
        proc, model = self._mock_blip("a red bus")
        caption, err = run_blip_caption(_dummy_image(), proc, model)
        assert caption == "a red bus"

    def test_inference_error_returns_none_and_message(self):
        proc = MagicMock()
        proc.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
        proc.side_effect = RuntimeError("BLIP OOM")
        model = MagicMock()
        caption, err = run_blip_caption(_dummy_image(), proc, model)
        assert caption is None
        assert err is not None
        assert "BLIP inference error" in err

    def test_model_generate_error_returns_none(self):
        proc = MagicMock()
        proc.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
        model = MagicMock()
        model.generate.side_effect = RuntimeError("generate failed")
        caption, err = run_blip_caption(_dummy_image(), proc, model)
        assert caption is None
        assert err is not None
