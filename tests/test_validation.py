"""
Tests for src/validation.py — no model dependencies.
"""

import io
import struct

import pytest
from PIL import Image

from src.validation import (
    MAX_UPLOAD_BYTES,
    validate_and_decode_image,
    validate_claim,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_png(width: int = 10, height: int = 10) -> bytes:
    """Return valid PNG bytes of the given dimensions."""
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg(width: int = 10, height: int = 10) -> bytes:
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_webp(width: int = 10, height: int = 10) -> bytes:
    img = Image.new("RGB", (width, height), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()


# ── validate_and_decode_image ────────────────────────────────────────

class TestValidateAndDecodeImage:
    def test_valid_png_returns_image(self):
        data = _make_png()
        image, err = validate_and_decode_image(data)
        assert err is None
        assert image is not None
        assert image.mode == "RGB"

    def test_valid_jpeg_returns_image(self):
        data = _make_jpeg()
        image, err = validate_and_decode_image(data)
        assert err is None
        assert image is not None

    def test_valid_webp_returns_image(self):
        data = _make_webp()
        image, err = validate_and_decode_image(data)
        assert err is None
        assert image is not None

    def test_empty_bytes_fails(self):
        image, err = validate_and_decode_image(b"")
        assert image is None
        assert err is not None
        assert "empty" in err.lower()

    def test_oversized_file_fails(self):
        # Create data that is larger than the limit.
        oversized = b"\x00" * (MAX_UPLOAD_BYTES + 1)
        image, err = validate_and_decode_image(oversized)
        assert image is None
        assert err is not None
        assert "10 MB" in err or "mb" in err.lower()

    def test_unsupported_format_fails(self):
        # Save a GIF (unsupported).
        img = Image.new("RGB", (10, 10))
        buf = io.BytesIO()
        img.save(buf, format="GIF")
        data = buf.getvalue()
        image, err = validate_and_decode_image(data)
        assert image is None
        assert err is not None
        assert "Unsupported" in err or "unsupported" in err.lower()

    def test_malformed_bytes_fails(self):
        image, err = validate_and_decode_image(b"\x89PNG not really valid garbage")
        assert image is None
        assert err is not None

    def test_random_non_image_bytes_fails(self):
        image, err = validate_and_decode_image(b"this is plain text, not an image")
        assert image is None
        assert err is not None

    def test_result_mode_is_rgb(self):
        """RGBA images (e.g. PNG with alpha) should be converted to RGB."""
        img = Image.new("RGBA", (20, 20), color=(0, 0, 255, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image, err = validate_and_decode_image(buf.getvalue())
        assert err is None
        assert image.mode == "RGB"

    def test_large_but_within_limit_succeeds(self):
        # A valid 1600×1600 PNG should be accepted (thumbnail'd).
        data = _make_png(800, 800)
        image, err = validate_and_decode_image(data)
        assert err is None
        assert image is not None


# ── validate_claim ───────────────────────────────────────────────────

class TestValidateClaim:
    def test_valid_claim(self):
        claim, err = validate_claim("A dog is running on grass.")
        assert err is None
        assert claim == "A dog is running on grass."

    def test_strips_whitespace(self):
        claim, err = validate_claim("  Hello world.  ")
        assert err is None
        assert claim == "Hello world."

    def test_blank_string_fails(self):
        claim, err = validate_claim("   ")
        assert claim is None
        assert err is not None
        assert "empty" in err.lower()

    def test_empty_string_fails(self):
        claim, err = validate_claim("")
        assert claim is None
        assert err is not None

    def test_non_string_fails(self):
        claim, err = validate_claim(None)  # type: ignore[arg-type]
        assert claim is None
        assert err is not None
