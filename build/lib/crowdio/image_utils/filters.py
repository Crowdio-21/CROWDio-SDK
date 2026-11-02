"""
Pre-built image filters / preprocessing functions.

Developers can pass these directly to distributed CrowdIO tasks or use
:func:`apply_filter` as a one-stop helper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


# ── Filter registry ─────────────────────────────────────────────────

AVAILABLE_FILTERS: list[str] = [
    "sharpen",
    "blur",
    "edge",
    "enhance",
    "grayscale",
    "emboss",
    "smooth",
    "contour",
    "invert",
]
"""List of filter names recognised by :func:`apply_filter`."""


def apply_filter(
    image: "Image.Image",
    filter_type: str,
    *,
    intensity: float = 1.0,
) -> "Image.Image":
    """
    Apply a named filter to a PIL Image.

    Args:
        image: Source PIL Image (will **not** be modified in place).
        filter_type: One of :data:`AVAILABLE_FILTERS`.
        intensity: Strength multiplier for filters that support it
                   (``enhance``, ``blur``).

    Returns:
        A new PIL Image with the filter applied.

    Raises:
        ValueError: If *filter_type* is not recognised.
    """
    from PIL import ImageFilter, ImageEnhance, ImageOps

    ft = filter_type.lower().strip()

    if ft == "sharpen":
        return image.filter(ImageFilter.SHARPEN)

    if ft == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=2 * intensity))

    if ft == "edge":
        return image.filter(ImageFilter.FIND_EDGES)

    if ft == "enhance":
        img = ImageEnhance.Contrast(image).enhance(1.0 + 0.5 * intensity)
        img = ImageEnhance.Brightness(img).enhance(1.0 + 0.2 * intensity)
        return img

    if ft == "grayscale":
        return image.convert("L").convert("RGB")

    if ft == "emboss":
        return image.filter(ImageFilter.EMBOSS)

    if ft == "smooth":
        return image.filter(ImageFilter.SMOOTH_MORE)

    if ft == "contour":
        return image.filter(ImageFilter.CONTOUR)

    if ft == "invert":
        return ImageOps.invert(image.convert("RGB"))

    raise ValueError(
        f"Unknown filter_type {filter_type!r}. "
        f"Available filters: {', '.join(AVAILABLE_FILTERS)}"
    )
