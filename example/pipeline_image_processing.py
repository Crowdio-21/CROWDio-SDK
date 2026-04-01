#!/usr/bin/env python3
"""
Pipeline Image Processing Test – Dependency Counter Validation
===============================================================

This test exercises the **dependency counter / pipeline execution** system
using a real image-processing workload.  Unlike the existing embarrassingly-
parallel tests that split, distribute, and reassemble on the *client* side,
this script pushes the **entire workflow** through a three-stage pipeline
where each stage depends on the previous one:

    Stage 0  (preprocess)  – Load image, split into tiles, encode to base64
    Stage 1  (process)     – Apply a filter to each tile  [receives upstream]
    Stage 2  (postprocess) – Reassemble filtered tiles    [receives upstream]

The dependency counter ensures that:
  * Stage 1 tasks remain "blocked" until ALL stage-0 tasks complete.
  * Stage 2 tasks remain "blocked" until ALL stage-1 tasks complete.
  * Only the final-stage results are returned to the caller.

Usage:
    # Start foreman + at least one worker first, then:
    uv run python tests/image_processing/pipeline_image_processing.py

    # With options:
    uv run python tests/image_processing/pipeline_image_processing.py \
        --image tests/image_processing/archive/dataset/sample.jpg \
        --tile-size 200 --filter sharpen --output ./pipeline_output

    # Process multiple images from a dataset directory:
    uv run python tests/image_processing/pipeline_image_processing.py \
        --dataset-dir tests/image_processing/archive/dataset \
        --percent 5 --tile-size 128 --filter enhance
"""

import sys
import os
import asyncio
import time
import math
import argparse

# Add project root to path so imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from developer_sdk import crowdio_connect, crowdio_disconnect, CROWDio, crowdio_pipeline

@CROWDio.task(
    checkpoint=True,
    checkpoint_interval=3.0,
    checkpoint_state=[
        "processed_count",
        "total_tiles",
        "avg_time",
        "progress",
        "processed_results",
    ],
)

# =====================================================================
# Stage 0 – Preprocess: load image and split into tiles
# =====================================================================

@CROWDio.task()
def preprocess_image(image_input):
    """
    Load an image from disk (or accept raw base64) and split it into tiles.

    Args:
        image_input: dict with keys:
            - image_path: (str) path to image file on the machine running
              this task.  **Because workers execute remotely, the image is
              base64-encoded before being sent.**
            - image_b64: (str) base64-encoded image bytes (alternative to
              image_path – used when the client pre-encodes).
            - tile_size: (int) tile width/height in pixels (default 200).
            - filter_type: (str) filter to apply later (passed through).
            - image_id: (str|int) identifier for this image.

    Returns:
        dict with:
            - image_id, original_size, tile_size, filter_type
            - tiles: list of tile dicts (tile_id, image b64, position, size)
    """
    from PIL import Image
    import base64
    import io
    import time

    start = time.time()
    image_id = image_input.get("image_id", "unknown")
    tile_size = image_input.get("tile_size", 200)
    filter_type = image_input.get("filter_type", "sharpen")

    # Decode the image from base64
    image_bytes = base64.b64decode(image_input["image_b64"])
    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size

    # Split into tiles
    tiles = []
    tile_id = 0
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile_w = min(tile_size, width - x)
            tile_h = min(tile_size, height - y)
            tile_img = img.crop((x, y, x + tile_w, y + tile_h))

            buf = io.BytesIO()
            tile_img.save(buf, format="PNG")
            tile_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            tiles.append({
                "tile_id": tile_id,
                "image": tile_b64,
                "position": [x, y],
                "size": [tile_w, tile_h],
            })
            tile_id += 1

    elapsed = time.time() - start
    print(
        f"[preprocess] image_id={image_id} size={width}x{height} "
        f"tiles={len(tiles)} tile_size={tile_size} time={elapsed:.3f}s"
    )

    return {
        "image_id": image_id,
        "original_size": [width, height],
        "tile_size": tile_size,
        "filter_type": filter_type,
        "tiles": tiles,
        "preprocess_time": elapsed,
    }


# =====================================================================
# Stage 1 – Process: apply filter to each tile
# =====================================================================

@CROWDio.task()
def process_tiles(task_input):
    """
    Apply an image filter to every tile received from the preprocess stage.

    Because this stage has ``pass_upstream_results=True``, the system
    injects upstream results into the arguments automatically:

        task_input = {
            "original_args": <whatever was in args_list>,
            "upstream_results": {
                "<upstream_task_id>": { ... preprocess result dict ... }
            }
        }

    Returns:
        dict with:
            - image_id, original_size, filter_type
            - processed_tiles: list of tile dicts with filtered image b64
    """
    from PIL import Image, ImageFilter, ImageEnhance
    import base64
    import io
    import time

    start = time.time()

    # Unpack upstream injection
    upstream = task_input.get("upstream_results", {})
    if not upstream:
        return {"error": "No upstream results received", "input_keys": list(task_input.keys())}

    # Combine tiles from all upstream preprocess tasks
    all_tiles = []
    image_id = "unknown"
    original_size = None
    filter_type = "sharpen"

    for _task_id, preprocess_result in upstream.items():
        # Handle result that may be a string (JSON) or already a dict
        if isinstance(preprocess_result, str):
            import json
            try:
                preprocess_result = json.loads(preprocess_result)
            except (json.JSONDecodeError, TypeError):
                import ast
                preprocess_result = ast.literal_eval(preprocess_result)

        image_id = preprocess_result.get("image_id", image_id)
        original_size = preprocess_result.get("original_size", original_size)
        filter_type = preprocess_result.get("filter_type", filter_type)
        all_tiles.extend(preprocess_result.get("tiles", []))

    # Apply the filter to each tile
    processed_tiles = []
    for tile_info in all_tiles:
        tile_bytes = base64.b64decode(tile_info["image"])
        tile_img = Image.open(io.BytesIO(tile_bytes))

        if filter_type == "sharpen":
            tile_img = tile_img.filter(ImageFilter.SHARPEN)
        elif filter_type == "blur":
            tile_img = tile_img.filter(ImageFilter.GaussianBlur(radius=2))
        elif filter_type == "edge":
            tile_img = tile_img.filter(ImageFilter.FIND_EDGES)
        elif filter_type == "enhance":
            enhancer = ImageEnhance.Contrast(tile_img)
            tile_img = enhancer.enhance(1.5)
            enhancer = ImageEnhance.Brightness(tile_img)
            tile_img = enhancer.enhance(1.2)
        elif filter_type == "grayscale":
            tile_img = tile_img.convert("L").convert("RGB")
        elif filter_type == "emboss":
            tile_img = tile_img.filter(ImageFilter.EMBOSS)
        elif filter_type == "smooth":
            tile_img = tile_img.filter(ImageFilter.SMOOTH_MORE)
        elif filter_type == "contour":
            tile_img = tile_img.filter(ImageFilter.CONTOUR)

        buf = io.BytesIO()
        tile_img.save(buf, format="PNG")
        filtered_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        processed_tiles.append({
            "tile_id": tile_info["tile_id"],
            "image": filtered_b64,
            "position": tile_info["position"],
            "size": tile_info["size"],
            "filter_applied": filter_type,
        })

    elapsed = time.time() - start
    print(
        f"[process] image_id={image_id} tiles_processed={len(processed_tiles)} "
        f"filter={filter_type} time={elapsed:.3f}s"
    )

    return {
        "image_id": image_id,
        "original_size": original_size,
        "filter_type": filter_type,
        "processed_tiles": processed_tiles,
        "process_time": elapsed,
    }


# =====================================================================
# Stage 2 – Postprocess: reassemble tiles into final image
# =====================================================================

@CROWDio.task()
def postprocess_image(task_input):
    """
    Reassemble processed tiles back into a complete image.

    Receives upstream results from ALL stage-1 (process) tasks and
    rebuilds the final image.

    Returns:
        dict with:
            - image_id, original_size, filter_applied
            - result_image: base64 of the reassembled image
            - stats: tile count, total processing time
    """
    from PIL import Image
    import base64
    import io
    import time

    start = time.time()

    # Unpack upstream injection
    upstream = task_input.get("upstream_results", {})
    if not upstream:
        return {"error": "No upstream results received", "input_keys": list(task_input.keys())}

    # Collect all processed tiles from upstream processing tasks
    all_processed_tiles = []
    image_id = "unknown"
    original_size = None
    filter_type = "sharpen"
    total_process_time = 0.0

    for _task_id, process_result in upstream.items():
        if isinstance(process_result, str):
            import json
            try:
                process_result = json.loads(process_result)
            except (json.JSONDecodeError, TypeError):
                import ast
                process_result = ast.literal_eval(process_result)

        image_id = process_result.get("image_id", image_id)
        original_size = process_result.get("original_size", original_size)
        filter_type = process_result.get("filter_type", filter_type)
        total_process_time += process_result.get("process_time", 0.0)
        all_processed_tiles.extend(process_result.get("processed_tiles", []))

    if not all_processed_tiles or not original_size:
        return {
            "error": "No processed tiles received or missing original_size",
            "image_id": image_id,
            "upstream_keys": list(upstream.keys()),
        }

    # Reassemble the image
    width, height = original_size
    result_img = Image.new("RGB", (width, height))
    sorted_tiles = sorted(all_processed_tiles, key=lambda t: t["tile_id"])

    for tile_data in sorted_tiles:
        tile_bytes = base64.b64decode(tile_data["image"])
        tile_img = Image.open(io.BytesIO(tile_bytes))
        position = tuple(tile_data["position"])
        result_img.paste(tile_img, position)

    # Encode the final image as base64
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    result_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    elapsed = time.time() - start
    print(
        f"[postprocess] image_id={image_id} tiles={len(sorted_tiles)} "
        f"output_size={width}x{height} reassembly_time={elapsed:.3f}s"
    )

    return {
        "image_id": image_id,
        "original_size": original_size,
        "filter_applied": filter_type,
        "result_image": result_b64,
        "stats": {
            "total_tiles": len(sorted_tiles),
            "preprocess_time": None,  # available in stage-0 result
            "process_time": total_process_time,
            "postprocess_time": elapsed,
        },
    }


# =====================================================================
# CLI helpers
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the pipeline dependency counter with image processing."
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--image",
        default=None,
        help="Path to a single image to process.",
    )
    grp.add_argument(
        "--dataset-dir",
        default=None,
        help="Path to dataset directory containing image files / class folders.",
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=10.0,
        help="Percentage of dataset images to process (default: 10%%).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=200,
        help="Tile size in pixels (default: 200).",
    )
    parser.add_argument(
        "--filter",
        default="sharpen",
        choices=[
            "sharpen", "blur", "edge", "enhance",
            "grayscale", "emboss", "smooth", "contour",
        ],
        help="Filter to apply (default: sharpen).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "pipeline_output"),
        help="Output directory for reassembled images.",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Foreman host (default: localhost).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Foreman WebSocket port (default: 9000).",
    )
    return parser.parse_args()


def collect_images(dataset_dir):
    """Recursively collect image paths from a directory tree."""
    image_paths = []
    for root, _, files in os.walk(dataset_dir):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                image_paths.append(os.path.join(root, filename))
    image_paths.sort()
    return image_paths


def generate_test_image(width=640, height=480):
    """
    Generate a synthetic test image if no real image is provided.
    Returns the image as base64-encoded PNG bytes.
    """
    from PIL import Image, ImageDraw, ImageFont
    import base64
    import io
    import random

    img = Image.new("RGB", (width, height), color=(30, 30, 60))
    draw = ImageDraw.Draw(img)

    # Draw a gradient background
    for y in range(height):
        r = int(30 + (y / height) * 100)
        g = int(30 + (y / height) * 50)
        b = int(60 + (y / height) * 120)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Draw some shapes to make the image interesting
    random.seed(42)  # reproducible
    for _ in range(15):
        x1 = random.randint(0, width - 80)
        y1 = random.randint(0, height - 80)
        x2 = x1 + random.randint(30, 80)
        y2 = y1 + random.randint(30, 80)
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        shape = random.choice(["rect", "ellipse"])
        if shape == "rect":
            draw.rectangle([x1, y1, x2, y2], fill=color, outline="white")
        else:
            draw.ellipse([x1, y1, x2, y2], fill=color, outline="white")

    # Add text
    draw.text(
        (width // 2 - 120, height // 2 - 10),
        "CROWDio Pipeline Test",
        fill="white",
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64, (width, height)


# =====================================================================
# Main – orchestrate the pipeline
# =====================================================================

async def run_pipeline_single(image_b64, image_id, tile_size, filter_type, label=""):
    """
    Run the 3-stage pipeline for a single image.

    Returns the final-stage result dict.
    """
    print(f"\n{'='*60}")
    print(f"Pipeline: Processing image {label or image_id}")
    print(f"  Tile size : {tile_size}px")
    print(f"  Filter    : {filter_type}")
    print(f"  Stages    : preprocess → process → postprocess")
    print(f"{'='*60}")

    stage_0_input = {
        "image_b64": image_b64,
        "tile_size": tile_size,
        "filter_type": filter_type,
        "image_id": image_id,
    }

    start = time.time()

    # The pipeline call – the dependency counter system handles the
    # stage-to-stage data flow and blocking automatically.
    results = await pipeline([
        {
            "func": preprocess_image,
            "args_list": [stage_0_input],       # 1 task in stage 0
            "name": "preprocess",
        },
        {
            "func": process_tiles,
            "args_list": [None],                # placeholder; upstream results injected
            "pass_upstream_results": True,
            "name": "process",
        },
        {
            "func": postprocess_image,
            "args_list": [None],                # placeholder; upstream results injected
            "pass_upstream_results": True,
            "name": "postprocess",
        },
    ])

    elapsed = time.time() - start

    print(f"\nPipeline completed in {elapsed:.2f}s")

    if results and len(results) > 0:
        final = results[0]
        if isinstance(final, str):
            import json as _json
            try:
                final = _json.loads(final)
            except Exception:
                import ast
                final = ast.literal_eval(final)

        if "error" in final:
            print(f"  ERROR: {final['error']}")
        else:
            stats = final.get("stats", {})
            print(f"  Image ID      : {final.get('image_id')}")
            print(f"  Output size   : {final.get('original_size')}")
            print(f"  Filter        : {final.get('filter_applied')}")
            print(f"  Total tiles   : {stats.get('total_tiles')}")
            print(f"  Process time  : {stats.get('process_time', 0):.3f}s")
            print(f"  Postprocess   : {stats.get('postprocess_time', 0):.3f}s")
            print(f"  Wall time     : {elapsed:.2f}s")

        return final, elapsed

    print("  No results returned!")
    return None, elapsed


async def main():
    args = parse_args()

    # ── Determine image inputs ──────────────────────────────────────
    import base64

    images = []  # list of (b64, image_id, label)

    if args.image:
        # Single image from disk
        if not os.path.isfile(args.image):
            raise FileNotFoundError(f"Image not found: {args.image}")
        with open(args.image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        images.append((b64, 0, os.path.basename(args.image)))

    elif args.dataset_dir:
        # Dataset directory
        if not os.path.isdir(args.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")
        paths = collect_images(args.dataset_dir)
        if not paths:
            raise ValueError(f"No images found in: {args.dataset_dir}")
        n = max(1, int(math.ceil(len(paths) * (args.percent / 100.0))))
        selected = paths[:n]
        print(f"Selected {len(selected)} of {len(paths)} images ({args.percent}%)")
        for idx, p in enumerate(selected):
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            images.append((b64, idx, os.path.basename(p)))

    else:
        # No image supplied – generate a synthetic test image
        print("No image or dataset provided — generating a synthetic test image.")
        b64, size = generate_test_image(640, 480)
        images.append((b64, 0, f"synthetic_{size[0]}x{size[1]}"))

    # ── Connect to foreman ──────────────────────────────────────────
    await crowdio_connect(args.host, args.port)

    try:
        overall_start = time.time()
        results_summary = []

        for img_b64, img_id, label in images:
            result, elapsed = await run_pipeline_single(
                image_b64=img_b64,
                image_id=img_id,
                tile_size=args.tile_size,
                filter_type=args.filter,
                label=label,
            )

            # Save output image if result contains one
            if result and "result_image" in result:
                os.makedirs(args.output_dir, exist_ok=True)
                out_name = f"pipeline_{args.filter}_{label}.png"
                out_path = os.path.join(args.output_dir, out_name)
                img_bytes = base64.b64decode(result["result_image"])
                with open(out_path, "wb") as f:
                    f.write(img_bytes)
                print(f"  Saved: {out_path}")
                results_summary.append({
                    "label": label,
                    "output": out_path,
                    "wall_time": elapsed,
                })
            else:
                results_summary.append({
                    "label": label,
                    "output": None,
                    "wall_time": elapsed,
                    "error": result.get("error") if result else "no result",
                })

        overall_time = time.time() - overall_start

        # ── Summary ─────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"PIPELINE TEST SUMMARY")
        print(f"{'='*60}")
        print(f"  Images processed : {len(images)}")
        print(f"  Tile size        : {args.tile_size}px")
        print(f"  Filter           : {args.filter}")
        print(f"  Total wall time  : {overall_time:.2f}s")
        successes = [r for r in results_summary if r.get("output")]
        failures = [r for r in results_summary if not r.get("output")]
        print(f"  Succeeded        : {len(successes)}")
        print(f"  Failed           : {len(failures)}")
        if failures:
            for f in failures:
                print(f"    FAIL: {f['label']} — {f.get('error', 'unknown')}")
        print(f"{'='*60}")

    finally:
        await crowdio_disconnect()


if __name__ == "__main__":
    asyncio.run(main())
