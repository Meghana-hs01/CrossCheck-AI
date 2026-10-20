"""
CrossCheck AI – model loading.

All pretrained models are loaded here and cached with st.cache_resource
(or functools.lru_cache when called outside Streamlit).  Every loader
returns (model_object, None) on success or (None, error_message) on
failure so callers can degrade gracefully without raising.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"

# -------------------------------------------------------------------
# CLIP
# -------------------------------------------------------------------

def _load_clip_impl() -> Tuple[object, object, Optional[str]]:
    """
    Returns (processor, model, error_message).
    error_message is None on success.
    """
    try:
        from transformers import CLIPModel, CLIPProcessor  # lazy import

        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        model = CLIPModel.from_pretrained(
            CLIP_MODEL_ID,
            use_safetensors=True,
        )
        model.eval()
        logger.info("CLIP model loaded: %s", CLIP_MODEL_ID)
        return processor, model, None
    except Exception as exc:  # noqa: BLE001
        msg = f"CLIP load failed: {type(exc).__name__}: {exc}"
        logger.error(msg)
        return None, None, msg


# Module-level singleton (used by non-Streamlit callers, e.g. pytest).
_clip_singleton: Optional[tuple] = None


def get_clip(use_streamlit_cache: bool = True):
    """
    Return (processor, model, error_message).

    When use_streamlit_cache=True this delegates to the Streamlit-cached
    version defined in app.py.  When False (tests, CLI) it uses a
    module-level singleton so models are not reloaded per test.
    """
    global _clip_singleton
    if _clip_singleton is None:
        _clip_singleton = _load_clip_impl()
    return _clip_singleton


# -------------------------------------------------------------------
# BLIP
# -------------------------------------------------------------------

def _load_blip_impl() -> Tuple[object, object, Optional[str]]:
    """
    Returns (processor, model, error_message).
    """
    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor  # lazy

        processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
        model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID)
        model.eval()
        logger.info("BLIP model loaded: %s", BLIP_MODEL_ID)
        return processor, model, None
    except Exception as exc:  # noqa: BLE001
        msg = f"BLIP load failed: {type(exc).__name__}: {exc}"
        logger.error(msg)
        return None, None, msg


_blip_singleton: Optional[tuple] = None


def get_blip(use_streamlit_cache: bool = True):
    """
    Return (processor, model, error_message).
    """
    global _blip_singleton
    if _blip_singleton is None:
        _blip_singleton = _load_blip_impl()
    return _blip_singleton
