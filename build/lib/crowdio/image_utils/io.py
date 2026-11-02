"""
Convenience I/O helpers for loading / saving images.
"""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


def load_image(path: str) -> "Image.Image":
    """
    Load an image from disk.

    Args:
        path: File path to the image.

    Returns:
        PIL Image object.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    from PIL import Image as _Image

    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return _Image.open(path)


def save_image(
    image: "Image.Image",
    path: str,
    *,
    fmt: str | None = None,
    makedirs: bool = True,
) -> str:
    """
    Save a PIL Image to disk.

    Args:
        image: PIL Image object.
        path: Destination file path.
        fmt: Explicit format (e.g. ``"PNG"``).  Inferred from extension
             if ``None``.
        makedirs: Create parent directories if they don't exist.

    Returns:
        Absolute path that was written.
    """
    path = os.path.abspath(path)
    if makedirs:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    save_kwargs: dict[str, Any] = {}
    if fmt:
        save_kwargs["format"] = fmt

    image.save(path, **save_kwargs)
    return path


def get_image_info(image: "Image.Image") -> dict[str, Any]:
    """
    Return a summary dict describing a PIL Image.

    Keys: ``width``, ``height``, ``mode``, ``format``, ``num_pixels``.
    """
    width, height = image.size
    return {
        "width": width,
        "height": height,
        "mode": image.mode,
        "format": image.format,
        "num_pixels": width * height,
    }
