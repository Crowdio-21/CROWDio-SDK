"""
Image chunking / splitting utilities.

Functions that partition a source image into smaller pieces suitable for
distribution across CrowdIO workers.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

from .encoding import encode_image


# ── Tile-based splitting ────────────────────────────────────────────


def split_image_into_tiles(
    image: "Image.Image",
    tile_size: int = 200,
    *,
    overlap: int = 0,
    fmt: str = "PNG",
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Split an image into square tiles.

    Args:
        image: PIL Image object.
        tile_size: Width/height of each tile in pixels.
        overlap: Number of overlapping pixels between adjacent tiles.
                 Useful for filters that need context at tile borders.
        fmt: Image format used when encoding tiles (``"PNG"``, ``"JPEG"``…).
        extra: Optional dict of additional metadata to attach to every tile.

    Returns:
        List of tile dicts, each containing::

            {
                "tile_id": int,
                "image": str,        # base64-encoded
                "position": (x, y),
                "size": (w, h),
                ...extra
            }
    """
    width, height = image.size
    step = max(1, tile_size - overlap)
    tiles: list[dict[str, Any]] = []
    tile_id = 0

    for y in range(0, height, step):
        for x in range(0, width, step):
            tile_w = min(tile_size, width - x)
            tile_h = min(tile_size, height - y)

            tile_img = image.crop((x, y, x + tile_w, y + tile_h))
            tile_b64 = encode_image(tile_img, fmt=fmt)

            tile_dict: dict[str, Any] = {
                "tile_id": tile_id,
                "image": tile_b64,
                "position": (x, y),
                "size": (tile_w, tile_h),
            }
            if extra:
                tile_dict.update(extra)

            tiles.append(tile_dict)
            tile_id += 1

    return tiles


# ── Grid-based splitting ────────────────────────────────────────────


def split_image_into_grid(
    image: "Image.Image",
    rows: int = 2,
    cols: int = 2,
    *,
    fmt: str = "PNG",
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Split an image into a fixed *rows × cols* grid.

    Unlike :func:`split_image_into_tiles`, this guarantees an exact number
    of pieces regardless of the image dimensions.

    Args:
        image: PIL Image object.
        rows: Number of horizontal slices.
        cols: Number of vertical slices.
        fmt: Encoding format.
        extra: Optional extra metadata per tile.

    Returns:
        List of tile dicts (same schema as :func:`split_image_into_tiles`).
    """
    width, height = image.size
    tile_w = width // cols
    tile_h = height // rows
    tiles: list[dict[str, Any]] = []
    tile_id = 0

    for r in range(rows):
        for c in range(cols):
            x = c * tile_w
            y = r * tile_h
            # Last column/row absorbs remaining pixels
            w = width - x if c == cols - 1 else tile_w
            h = height - y if r == rows - 1 else tile_h

            tile_img = image.crop((x, y, x + w, y + h))
            tile_b64 = encode_image(tile_img, fmt=fmt)

            tile_dict: dict[str, Any] = {
                "tile_id": tile_id,
                "image": tile_b64,
                "position": (x, y),
                "size": (w, h),
            }
            if extra:
                tile_dict.update(extra)

            tiles.append(tile_dict)
            tile_id += 1

    return tiles


# ── Strip-based splitting ───────────────────────────────────────────


def split_image_into_strips(
    image: "Image.Image",
    num_strips: int = 4,
    *,
    direction: str = "horizontal",
    fmt: str = "PNG",
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Split an image into horizontal or vertical strips.

    Args:
        image: PIL Image object.
        num_strips: Number of strips.
        direction: ``"horizontal"`` (default) or ``"vertical"``.
        fmt: Encoding format.
        extra: Optional extra metadata per strip.

    Returns:
        List of strip dicts with the same schema as tiles.
    """
    width, height = image.size
    strips: list[dict[str, Any]] = []

    if direction == "horizontal":
        strip_h = height // num_strips
        for i in range(num_strips):
            y = i * strip_h
            h = height - y if i == num_strips - 1 else strip_h
            strip_img = image.crop((0, y, width, y + h))
            strips.append(
                {
                    "tile_id": i,
                    "image": encode_image(strip_img, fmt=fmt),
                    "position": (0, y),
                    "size": (width, h),
                    **(extra or {}),
                }
            )
    elif direction == "vertical":
        strip_w = width // num_strips
        for i in range(num_strips):
            x = i * strip_w
            w = width - x if i == num_strips - 1 else strip_w
            strip_img = image.crop((x, 0, x + w, height))
            strips.append(
                {
                    "tile_id": i,
                    "image": encode_image(strip_img, fmt=fmt),
                    "position": (x, 0),
                    "size": (w, height),
                    **(extra or {}),
                }
            )
    else:
        raise ValueError(
            f"direction must be 'horizontal' or 'vertical', got {direction!r}"
        )

    return strips
