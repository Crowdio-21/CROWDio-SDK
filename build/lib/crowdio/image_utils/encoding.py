"""
Image encoding / decoding utilities for base64 serialization.

These helpers handle the conversion between PIL Image objects and
base64 strings, which is the transport format used by CrowdIO tasks.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


def encode_image(image: "Image.Image", fmt: str = "PNG") -> str:
    """
    Encode a PIL Image to a base64 string.

    Args:
        image: PIL Image object.
        fmt: Image format for encoding (e.g. ``"PNG"``, ``"JPEG"``).

    Returns:
        Base64-encoded string of the image bytes.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_image(b64_string: str) -> "Image.Image":
    """
    Decode a base64 string back into a PIL Image.

    Args:
        b64_string: Base64-encoded image data.

    Returns:
        PIL Image object.
    """
    from PIL import Image as _Image

    image_bytes = base64.b64decode(b64_string)
    return _Image.open(io.BytesIO(image_bytes))
