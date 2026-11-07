"""
CrossCheck AI – image validation utilities.
"""

from __future__ import annotations

import io
import warnings
from typing import Tuple, Optional

from PIL import Image, ImageOps

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_PIXELS = 20_000_000

# Apply the pixel cap globally (guards against decompression bombs).
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def validate_and_decode_image(data: bytes) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Validate raw image bytes and return a decoded PIL Image.

    Returns (image, None) on success, (None, error_message) on failure.

    Checks performed:
    - Not empty.
    - Does not exceed MAX_UPLOAD_BYTES.
    - Format is one of SUPPORTED_FORMATS.
    - File is structurally valid (verify()).
    - Pixel count within MAX_PIXELS (DecompressionBombWarning → error).
    - Orientation metadata applied (EXIF transpose).
    - Converted to RGB for model compatibility.
    """
    if not data:
        return None, "The uploaded file is empty."

    if len(data) > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        return None, f"Upload an image no larger than {mb} MB."

    # First pass: verify format and structure without decoding pixels.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as src:
                fmt = src.format
                if fmt not in SUPPORTED_FORMATS:
                    supported = ", ".join(sorted(SUPPORTED_FORMATS))
                    return None, f"Unsupported format '{fmt}'. Supported: {supported}."
                src.verify()  # checks for file corruption
    except Image.DecompressionBombWarning:
        return None, (
            f"Image exceeds the maximum allowed pixel count ({MAX_PIXELS:,}). "
            "Please use a smaller image."
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"Image could not be decoded: {type(exc).__name__}: {exc}"

    # Second pass: decode pixels (verify() consumes the stream).
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as src:
                image = ImageOps.exif_transpose(src).convert("RGB")
                image.thumbnail((1600, 1600))
                return image, None
    except Image.DecompressionBombWarning:
        return None, (
            f"Image exceeds the maximum allowed pixel count ({MAX_PIXELS:,})."
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"Image decoding failed: {type(exc).__name__}: {exc}"


def validate_claim(claim: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Strip and validate the claim string.

    Returns (clean_claim, None) on success, (None, error_message) on failure.
    """
    if not isinstance(claim, str):
        return None, "Claim must be a text string."
    clean = claim.strip()
    if not clean:
        return None, "Claim is empty. Please enter an English caption or claim."
    return clean, None
