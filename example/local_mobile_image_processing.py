#!/usr/bin/env python3
"""
Simple mobile image processing example using Crowdio constants.

Goal:
- Developers write readable config using crowdio.Constant.FILE_DIR.
- Mobile runtime resolves constants to real device paths before execution.

Expected mobile runtime behavior:
- Inject a dict into builtins._crowdio_path_aliases, for example:
    {
            "@CROWDIO:FILE_DIR": "/storage/emulated/0/MyPickedFolder"
    }
"""

import argparse
import asyncio
import base64
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from crowdio import crowdio_connect, crowdio_disconnect, CROWDio, crowdio_map


@CROWDio.task(
    checkpoint=True, checkpoint_interval=3.0, checkpoint_state=["processed", "errors"]
)
def process_images_on_device(config):
    """Run on worker and return processed image previews back to the client."""
    import glob
    import builtins
    import platform
    from PIL import Image, ImageEnhance, ImageFilter
    import base64
    import io
    import os
    import time

    def resolve_path_alias(value):
        if not isinstance(value, str) or not value.startswith("@CROWDIO:"):
            return value
        alias_map = getattr(builtins, "_crowdio_path_aliases", {})
        return alias_map.get(value, value)

    def apply_filter(img, filter_name):
        if filter_name == "sharpen":
            return img.filter(ImageFilter.SHARPEN)
        if filter_name == "blur":
            return img.filter(ImageFilter.GaussianBlur(radius=2))
        if filter_name == "edge":
            return img.filter(ImageFilter.FIND_EDGES)
        if filter_name == "enhance":
            contrast = ImageEnhance.Contrast(img).enhance(1.4)
            return ImageEnhance.Brightness(contrast).enhance(1.15)
        if filter_name == "grayscale":
            return img.convert("L").convert("RGB")
        return img

    device_id = platform.node() or "unknown-device"

    image_dir = resolve_path_alias(config.get("image_dir"))
    filter_name = config.get("filter", "sharpen")
    max_images = int(config.get("max_images", 0))
    preview_quality = int(config.get("preview_quality", 70))

    if isinstance(image_dir, str) and image_dir.startswith("@CROWDIO:"):
        return {
            "device_id": device_id,
            "processed": 0,
            "items": [],
            "errors": [
                "Path alias was not resolved on worker. "
                "Ensure mobile runtime injects builtins._crowdio_path_aliases."
            ],
        }

    if not image_dir or not os.path.isdir(image_dir):
        return {
            "device_id": device_id,
            "processed": 0,
            "items": [],
            "errors": [f"Image directory not found: {image_dir}"],
        }

    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    image_paths = []
    for pattern in patterns:
        image_paths.extend(
            glob.glob(os.path.join(image_dir, "**", pattern), recursive=True)
        )
    image_paths.sort()

    if max_images > 0:
        image_paths = image_paths[:max_images]

    if not image_paths:
        return {
            "device_id": device_id,
            "processed": 0,
            "items": [],
            "errors": [f"No images found in: {image_dir}"],
        }

    start = time.time()
    processed = 0
    errors = []
    items = []

    for src_path in image_paths:
        try:
            image = Image.open(src_path).convert("RGB")
            result = apply_filter(image, filter_name)

            # Return a compact preview to client instead of writing to mobile storage.
            preview = result.copy()
            preview.thumbnail((256, 256))
            buf = io.BytesIO()
            preview.save(buf, format="JPEG", quality=preview_quality)

            items.append(
                {
                    "source_name": os.path.basename(src_path),
                    "size": list(image.size),
                    "preview_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
                }
            )

            processed += 1
        except Exception as exc:
            errors.append(f"{src_path}: {exc}")

    return {
        "device_id": device_id,
        "processed": processed,
        "items": items,
        "errors": errors,
        "filter": filter_name,
        "elapsed": round(time.time() - start, 3),
        "image_dir": image_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple mobile image processing using crowdio.Constant path aliases."
    )
    parser.add_argument(
        "filter",
        nargs="?",
        default="sharpen",
        choices=["sharpen", "blur", "edge", "enhance", "grayscale"],
        help="Filter to apply.",
    )
    parser.add_argument(
        "--max-images", type=int, default=10, help="Max images to process."
    )
    parser.add_argument(
        "--preview-quality",
        type=int,
        default=70,
        help="JPEG quality for previews returned to client (1-95).",
    )
    parser.add_argument(
        "--client-output-dir",
        default=os.path.join(
            os.path.dirname(__file__), "pipeline_output", "local_mobile_previews"
        ),
        help="Local directory on client machine to save returned preview images.",
    )
    parser.add_argument("--host", default="localhost", help="Foreman host.")
    parser.add_argument("--port", type=int, default=9000, help="Foreman port.")

    return parser.parse_args()


async def main():
    args = parse_args()

    # Developer-friendly config: no hardcoded Android paths.
    # Mobile runtime can resolve crowdio.Constant.FILE_DIR to selected folder path.
    task_config = {
        "image_dir": CROWDio.Constant.FILE_DIR,
        "filter": args.filter,
        "max_images": args.max_images,
        "preview_quality": args.preview_quality,
    }

    print("\n" + "=" * 64)
    print("Mobile Image Processing (Constant Alias Example)")
    print("=" * 64)
    print(f"image_dir alias : {task_config['image_dir']}")
    print(f"filter          : {task_config['filter']}")
    print(f"max_images      : {task_config['max_images']}")
    print(f"preview_quality : {task_config['preview_quality']}")
    print(f"client_output   : {args.client_output_dir}")
    print("=" * 64)

    await crowdio_connect(args.host, args.port)
    try:
        started = time.time()
        results = await crowdio_map(process_images_on_device, [task_config])
        wall_time = time.time() - started

        print("\nResults")
        print("-" * 64)
        os.makedirs(args.client_output_dir, exist_ok=True)
        saved_previews = 0
        for idx, result in enumerate(results):
            processed = result.get("processed", 0)
            errors = result.get("errors", [])
            items = result.get("items", [])

            for preview_index, item in enumerate(items):
                payload = item.get("preview_base64")
                if not payload:
                    continue
                try:
                    src_name = item.get("source_name", f"image_{preview_index}")
                    safe_name, _ = os.path.splitext(src_name)
                    out_name = f"worker{idx}_{preview_index}_{safe_name}.jpg"
                    out_path = os.path.join(args.client_output_dir, out_name)
                    with open(out_path, "wb") as fh:
                        fh.write(base64.b64decode(payload))
                    saved_previews += 1
                except Exception as exc:
                    errors.append(f"preview_save_error({preview_index}): {exc}")

            print(
                f"Worker {idx}: processed={processed}, previews={len(items)}, "
                f"errors={len(errors)}"
            )
        print("-" * 64)
        print(f"Saved previews: {saved_previews} -> {args.client_output_dir}")
        print(f"Wall time: {wall_time:.2f}s")
    finally:
        await crowdio_disconnect()


if __name__ == "__main__":
    asyncio.run(main())
