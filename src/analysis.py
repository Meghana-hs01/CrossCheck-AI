"""
CrossCheck AI – inference functions.

Each function accepts already-loaded model objects so they can be
called with real or mocked models in tests.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# CLIP supports up to 77 tokens (including BOS/EOS).
CLIP_MAX_TOKENS = 77


def run_clip(
    image: Image.Image,
    claim: str,
    processor,
    model,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Compute CLIP cosine similarity between *image* and *claim*.

    Returns (score: float, error: None) on success, or
            (None, error_message: str) on failure.

    Raises ValueError for token-limit violations so callers can
    surface the message without catching generic exceptions.
    """
    # Token-limit check (explicit, not silent truncation).
    token_ids = processor.tokenizer(
        claim,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    if len(token_ids) > CLIP_MAX_TOKENS:
        raise ValueError(
            f"Claim is too long ({len(token_ids)} tokens). "
            f"CLIP supports at most {CLIP_MAX_TOKENS} tokens "
            "including the start and end tokens. "
            "Please shorten your claim."
        )

    try:
        inputs = processor(
            text=[claim],
            images=image,
            return_tensors="pt",
            padding=True,
        )

        with torch.inference_mode():
            outputs = model(**inputs)
            score = torch.nn.functional.cosine_similarity(
                outputs.image_embeds.float(),
                outputs.text_embeds.float(),
                dim=-1,
            ).item()

        return score, None

    except Exception as exc:  # noqa: BLE001
        msg = f"CLIP inference error: {type(exc).__name__}: {exc}"
        logger.error(msg)
        return None, msg


def run_blip_caption(
    image: Image.Image,
    processor,
    model,
    max_new_tokens: int = 100,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate a caption for *image* using BLIP.

    Returns (caption: str, error: None) on success, or
            (None, error_message: str) on failure.

    The returned caption is a model-generated description and may
    hallucinate or omit visual details.
    """
    try:
        inputs = processor(image, return_tensors="pt")

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        logger.info("BLIP caption: %r", caption)
        return caption, None

    except Exception as exc:  # noqa: BLE001
        msg = f"BLIP inference error: {type(exc).__name__}: {exc}"
        logger.error(msg)
        return None, msg


def run_nli(
    premise: str,
    hypothesis: str,
    tokenizer,
    model,
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Run NLI inference with *premise* as the visual description and
    *hypothesis* as the user claim.

    Label mapping is read directly from model.config.id2label — never
    assumed or hardcoded.

    Returns (scores_dict, None) on success where scores_dict has keys
    'entailment', 'contradiction', 'neutral' (lowercase, float 0-1).
    Returns (None, error_message) on failure.

    These are textual-logic scores, NOT visual-truth probabilities.
    """
    try:
        id2label = getattr(model.config, "id2label", None)
        if not id2label:
            return None, "NLI model has no id2label mapping in config."

        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()

        if isinstance(probs, float):
            probs = [probs]

        scores = {}
        for idx, prob in enumerate(probs):
            label = id2label[idx].lower()
            scores[label] = round(float(prob), 4)

        logger.info("NLI scores: %s", scores)

        normalised = {
            "entailment": scores.get("entailment", 0.0),
            "contradiction": scores.get("contradiction", 0.0),
            "neutral": scores.get("neutral", 0.0),
        }
        return normalised, None

    except Exception as exc:  # noqa: BLE001
        msg = f"NLI inference error: {type(exc).__name__}: {exc}"
        logger.error(msg)
        return None, msg
