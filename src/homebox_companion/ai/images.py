"""Image encoding and optimization utilities for AI/LLM APIs."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from loguru import logger
from PIL import Image

# Default settings for image optimization (for AI vision)
DEFAULT_MAX_DIMENSION = 2048  # Most vision models work best with max 2048px images
DEFAULT_JPEG_QUALITY = 85

# PIL format to MIME type mapping
_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
}


def _detect_mime_type(image_bytes: bytes) -> str:
    """Detect the MIME type of an image from its bytes.

    Args:
        image_bytes: Raw image data.

    Returns:
        MIME type string (e.g., "image/jpeg", "image/png").
        Falls back to "image/jpeg" if detection fails.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        fmt = img.format
        if fmt and fmt.upper() in _FORMAT_TO_MIME:
            return _FORMAT_TO_MIME[fmt.upper()]
    except Exception:
        pass
    return "image/jpeg"  # Safe fallback for unknown formats


def _normalize_image(img: Image.Image) -> Image.Image:
    """Normalize image: handle EXIF orientation and convert to RGB.

    This is a shared helper for both optimize_image_for_vision() and
    compress_image_for_upload() to avoid code duplication.

    Args:
        img: PIL Image object to normalize.

    Returns:
        Normalized PIL Image in RGB mode with correct orientation.
    """
    # Handle EXIF orientation
    try:
        from PIL import ExifTags

        orientation: int | None = None
        for tag_id in ExifTags.TAGS:
            if ExifTags.TAGS[tag_id] == "Orientation":
                orientation = tag_id
                break

        exif = img.getexif()
        if exif is not None and orientation is not None:
            orientation_value = exif.get(orientation)
            if orientation_value == 3:
                img = img.rotate(180, expand=True)
            elif orientation_value == 6:
                img = img.rotate(270, expand=True)
            elif orientation_value == 8:
                img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, TypeError):
        # No EXIF data or no orientation tag
        pass

    # Convert to RGB if necessary (handles RGBA, P mode, etc.)
    if img.mode in ("RGBA", "P", "LA"):
        # Create white background for transparent images
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img


def optimize_image_for_vision(
    image_bytes: bytes,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> tuple[bytes, str]:
    """Optimize an image for LLM vision processing.

    Resizes large images and compresses to JPEG for faster uploads and
    reduced token costs. Vision models work best with images
    up to 2048px on the longest side.

    Args:
        image_bytes: Raw image data.
        max_dimension: Maximum width or height in pixels.
        quality: JPEG compression quality (1-100).

    Returns:
        Tuple of (optimized_bytes, mime_type).
    """
    original_size = len(image_bytes)

    try:
        img = Image.open(io.BytesIO(image_bytes))
        original_dimensions = img.size

        # Normalize image (EXIF orientation + RGB conversion)
        img = _normalize_image(img)

        # Resize if larger than max dimension
        needs_resize = max(img.size) > max_dimension
        if needs_resize:
            img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            logger.debug(f"Resized image from {original_dimensions} to {img.size}")

        # Compress to JPEG
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=quality, optimize=True)
        optimized_bytes = output.getvalue()

        optimized_size = len(optimized_bytes)
        savings = ((original_size - optimized_size) / original_size) * 100

        if savings > 5:  # Only log if meaningful savings
            logger.debug(f"Image optimized: {original_size:,} -> {optimized_size:,} bytes ({savings:.1f}% reduction)")

        return optimized_bytes, "image/jpeg"

    except Exception as e:
        logger.warning(f"Image optimization failed, using original: {e}")
        # Return original bytes with their actual MIME type, not hardcoded JPEG
        return image_bytes, _detect_mime_type(image_bytes)


def encode_image_to_data_uri(image_path: Path | str, optimize: bool = True) -> str:
    """Read an image file and return a data URI for vision model APIs.

    Args:
        image_path: Path to the image file.
        optimize: Whether to optimize the image for vision processing.

    Returns:
        A data URI string (e.g., "data:image/jpeg;base64,...").
    """
    path = Path(image_path)
    image_bytes = path.read_bytes()

    if optimize:
        image_bytes, mime_type = optimize_image_for_vision(image_bytes)
    else:
        mime_type = _detect_mime_type(image_bytes)

    # Extract suffix from mime_type (e.g., "image/jpeg" -> "jpeg")
    suffix = mime_type.split("/")[-1] if "/" in mime_type else "jpeg"
    payload = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/{suffix};base64,{payload}"


def encode_image_bytes_to_data_uri(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    optimize: bool = True,
    max_dimension: int | None = None,
    quality: int | None = None,
) -> str:
    """Encode raw image bytes to a data URI for vision model APIs.

    Args:
        image_bytes: Raw image data.
        mime_type: MIME type of the image.
        optimize: Whether to optimize the image for vision processing.
        max_dimension: Optional max dimension override for optimization.
        quality: Optional JPEG quality override for optimization.

    Returns:
        A data URI string.
    """
    if optimize:
        image_bytes, mime_type = optimize_image_for_vision(
            image_bytes,
            max_dimension=max_dimension or DEFAULT_MAX_DIMENSION,
            quality=quality if quality is not None else DEFAULT_JPEG_QUALITY,
        )

    # Extract suffix from mime_type (e.g., "image/jpeg" -> "jpeg")
    suffix = mime_type.split("/")[-1] if "/" in mime_type else "jpeg"
    payload = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/{suffix};base64,{payload}"


def compress_image_for_upload(
    image_bytes: bytes,
    max_dimension: int | None = None,
    quality: int = 75,
) -> tuple[bytes, str]:
    """Compress an image for Homebox upload based on quality settings.

    This function is separate from optimize_image_for_vision() to allow
    different compression strategies for AI (which needs good quality)
    vs Homebox storage (where users may prefer smaller files).

    Args:
        image_bytes: Raw image data.
        max_dimension: Maximum width or height in pixels. None = no resizing.
        quality: JPEG compression quality (1-100).

    Returns:
        Tuple of (compressed_bytes, mime_type).
    """
    # If raw quality requested (no max_dimension), return original with correct MIME type
    if max_dimension is None:
        return image_bytes, _detect_mime_type(image_bytes)

    original_size = len(image_bytes)

    try:
        img = Image.open(io.BytesIO(image_bytes))
        original_dimensions = img.size

        # Normalize image (EXIF orientation + RGB conversion)
        img = _normalize_image(img)

        # Resize if larger than max dimension
        needs_resize = max(img.size) > max_dimension
        if needs_resize:
            img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            logger.debug(f"Resized image for upload from {original_dimensions} to {img.size}")

        # Compress to JPEG
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=quality, optimize=True)
        compressed_bytes = output.getvalue()

        compressed_size = len(compressed_bytes)
        savings = ((original_size - compressed_size) / original_size) * 100

        if savings > 5:  # Only log if meaningful savings
            logger.debug(
                f"Image compressed for upload: {original_size:,} -> {compressed_size:,} bytes "
                f"({savings:.1f}% reduction)"
            )

        return compressed_bytes, "image/jpeg"

    except Exception as e:
        logger.warning(f"Image compression failed, using original: {e}")
        # Return original bytes with their actual MIME type, not hardcoded JPEG
        return image_bytes, _detect_mime_type(image_bytes)


def crop_image_bytes(
    image_bytes: bytes,
    bbox: list[float],
    padding_ratio: float = 0.08,
) -> bytes:
    """Crop an image to a normalized bounding box.

    The bbox is expressed as [x1, y1, x2, y2] with each value normalized
    to 0-1000 relative to the image dimensions (0 = top/left, 1000 =
    bottom/right). This is resolution-independent so the LLM can report
    boxes without knowing pixel dimensions.

    A small padding ratio is added around the box so the crop keeps a
    little context, and the box is clamped to the image bounds.

    Args:
        image_bytes: Raw image data.
        bbox: Normalized bounding box [x1, y1, x2, y2], values 0-1000.
        padding_ratio: Fraction of the box size to add as padding (default 8%).

    Returns:
        Cropped image bytes (original format). On any failure, returns
        the original bytes unchanged so callers can fall back gracefully.
    """
    if len(bbox) != 4:
        logger.warning(f"Invalid bbox length {len(bbox)}, returning original image")
        return image_bytes

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = _normalize_image(img)
        width, height = img.size

        # Denormalize 0-1000 coordinates to pixels
        x1 = max(0.0, min(bbox[0], bbox[2])) / 1000 * width
        y1 = max(0.0, min(bbox[1], bbox[3])) / 1000 * height
        x2 = min(1000.0, max(bbox[0], bbox[2])) / 1000 * width
        y2 = min(1000.0, max(bbox[1], bbox[3])) / 1000 * height

        # Add padding
        pad_x = (x2 - x1) * padding_ratio
        pad_y = (y2 - y1) * padding_ratio
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)

        # Ensure a sane minimum size (avoid degenerate 1px crops)
        if x2 - x1 < 8 or y2 - y1 < 8:
            logger.warning(f"Degenerate crop box {(x1, y1, x2, y2)}, returning original image")
            return image_bytes

        cropped = img.crop((int(x1), int(y1), int(x2), int(y2)))

        output = io.BytesIO()
        img_format = img.format or "JPEG"
        if img_format.upper() not in _FORMAT_TO_MIME:
            img_format = "JPEG"
        if img_format.upper() == "JPEG":
            cropped.save(output, format="JPEG", quality=90)
        else:
            cropped.save(output, format=img_format)
        return output.getvalue()

    except Exception as e:
        logger.warning(f"Image crop failed, using original: {e}")
        return image_bytes


def encode_compressed_image_to_base64(
    image_bytes: bytes,
    max_dimension: int | None = None,
    quality: int = 75,
) -> tuple[str, str]:
    """Compress an image and encode to base64 for API response.

    Args:
        image_bytes: Raw image data.
        max_dimension: Maximum width or height in pixels. None = no resizing.
        quality: JPEG compression quality (1-100).

    Returns:
        Tuple of (base64_string, mime_type).
    """
    compressed_bytes, mime_type = compress_image_for_upload(image_bytes, max_dimension, quality)
    base64_str = base64.b64encode(compressed_bytes).decode("ascii")
    return base64_str, mime_type
