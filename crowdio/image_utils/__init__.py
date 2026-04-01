"""
CrowdIO Image Utilities

Reusable image preprocessing, chunking, and aggregation helpers for
distributed image processing pipelines built on CrowdIO.

Usage:
    from crowdio.image_utils import (
        split_image_into_tiles,
        reassemble_tiles,
        apply_filter,
        encode_image,
        decode_image,
        load_image,
        save_image,
        get_image_info,
    )

    # Split → distribute → reassemble workflow
    tiles = split_image_into_tiles(image, tile_size=256)
    processed = await process_tile.map(tiles)
    result = reassemble_tiles(processed, image.size)
"""

from .chunking import (
    split_image_into_tiles,
    split_image_into_grid,
    split_image_into_strips,
)
from .aggregation import (
    reassemble_tiles,
    reassemble_strips,
    merge_results,
)
from .filters import (
    apply_filter,
    AVAILABLE_FILTERS,
)
from .encoding import (
    encode_image,
    decode_image,
)
from .io import (
    load_image,
    save_image,
    get_image_info,
)

__all__ = [
    # Chunking / splitting
    "split_image_into_tiles",
    "split_image_into_grid",
    "split_image_into_strips",
    # Aggregation / reassembly
    "reassemble_tiles",
    "reassemble_strips",
    "merge_results",
    # Filters / preprocessing
    "apply_filter",
    "AVAILABLE_FILTERS",
    # Encoding helpers
    "encode_image",
    "decode_image",
    # I/O helpers
    "load_image",
    "save_image",
    "get_image_info",
]
