"""Unit tests for crop_image_bytes in ai/images.py."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from homebox_companion.ai.images import crop_image_bytes

pytestmark = pytest.mark.unit


def _make_png(width: int = 100, height: int = 100, color: tuple = (255, 0, 0)) -> bytes:
    """Create a test PNG with a solid color."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestCropImageBytes:
    """Test normalized bbox cropping."""

    def test_crop_center_region(self) -> None:
        """A center bbox should produce a smaller image."""
        original = _make_png(100, 100)
        # bbox covering the center quarter: [250, 250, 750, 750]
        result = crop_image_bytes(original, [250, 250, 750, 750])

        img = Image.open(io.BytesIO(result))
        # 50% of 100px + 8% padding on each side ≈ 58px
        assert 40 < img.width < 70
        assert 40 < img.height < 70

    def test_full_image_bbox_returns_same_dimensions(self) -> None:
        """A full-frame bbox should roughly preserve dimensions."""
        original = _make_png(200, 100)
        result = crop_image_bytes(original, [0, 0, 1000, 1000], padding_ratio=0.0)

        img = Image.open(io.BytesIO(result))
        assert img.width == 200
        assert img.height == 100

    def test_out_of_range_values_are_clamped(self) -> None:
        """Coordinates outside 0-1000 should be clamped, not crash."""
        original = _make_png(100, 100)
        result = crop_image_bytes(original, [800, 800, 1200, 1200])

        assert len(result) > 0
        img = Image.open(io.BytesIO(result))
        assert img.width > 0
        assert img.height > 0

    def test_inverted_box_is_normalized(self) -> None:
        """Swapped corners (x2 < x1) should be handled gracefully."""
        original = _make_png(100, 100)
        result = crop_image_bytes(original, [750, 750, 250, 250])

        assert len(result) > 0
        img = Image.open(io.BytesIO(result))
        assert img.width > 0

    def test_invalid_bbox_length_returns_original(self) -> None:
        """A bbox without exactly 4 values falls back to the original."""
        original = _make_png(100, 100)
        result = crop_image_bytes(original, [0, 0, 1000])
        assert result == original

    def test_invalid_bytes_returns_original(self) -> None:
        """Garbage input falls back to the original bytes."""
        result = crop_image_bytes(b"not-an-image", [0, 0, 1000, 1000])
        assert result == b"not-an-image"

    def test_non_jpeg_source_format(self) -> None:
        """PNG sources should stay PNG after crop."""
        original = _make_png(100, 100, color=(0, 128, 255))
        result = crop_image_bytes(original, [100, 100, 900, 900])

        img = Image.open(io.BytesIO(result))
        assert img.format == "PNG"
