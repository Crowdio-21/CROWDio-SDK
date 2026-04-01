"""
Image aggregation / reassembly utilities.

Functions that recombine processed image pieces back into a complete image.
"""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

from .encoding import decode_image


def reassemble_tiles(
    processed_tiles: list[dict[str, Any]],
    original_size: tuple[int, int],
    *,
    mode: str = "RGB",
) -> "Image.Image":
    """
    Reassemble processed tiles back into a complete image.

    Args:
        processed_tiles: List of tile dicts. Each must contain at least
            ``"tile_id"``, ``"image"`` (base64), and ``"position"`` (x, y).
        original_size: ``(width, height)`` of the target image.
        mode: PIL image mode for the blank canvas (default ``"RGB"``).

    Returns:
        PIL Image with all tiles pasted at their original positions.
    """
    from PIL import Image as _Image

    result = _Image.new(mode, original_size)
    sorted_tiles = sorted(processed_tiles, key=lambda t: t["tile_id"])

    for tile_data in sorted_tiles:
        tile_img = decode_image(tile_data["image"])
        position = tuple(tile_data["position"])
        result.paste(tile_img, position)

    return result


# Alias – strips use the same layout logic as tiles.
reassemble_strips = reassemble_tiles
reassemble_strips.__doc__ = (
    "Alias for :func:`reassemble_tiles` — strips share the same reassembly logic."
)


def merge_results(
    results: list[dict[str, Any]],
    *,
    key: str = "tile_id",
    merge_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Merge / deduplicate a list of result dicts by a key.

    Useful when overlapping tiles produce duplicate regions or when
    multiple processing passes need to be combined.

    Args:
        results: Raw result dicts from workers.
        key: Field used as the unique identifier (default ``"tile_id"``).
        merge_fn: Optional callable that receives a list of dicts sharing
                  the same *key* value and returns a single merged dict.
                  If ``None`` the last dict wins.

    Returns:
        Deduplicated & merged list of result dicts, sorted by *key*.
    """
    from collections import defaultdict

    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[r[key]].append(r)

    merged: list[dict[str, Any]] = []
    for k in sorted(groups):
        items = groups[k]
        if merge_fn is not None:
            merged.append(merge_fn(items))
        else:
            merged.append(items[-1])

    return merged
