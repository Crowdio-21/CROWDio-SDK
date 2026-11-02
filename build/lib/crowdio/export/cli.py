"""``crowdio-export``: export verified ONNX shards and a manifest for one model."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .planner import (
    DEFAULT_BUDGET_RATIO,
    DEFAULT_EXTERNAL_DATA_THRESHOLD,
    DEFAULT_MAX_SHARD_BYTES,
    DEFAULT_PHONE_SHARD_BUDGET_BYTES,
    StorageBudgetExceeded,
    parse_ranges,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crowdio-export", description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model name or local path")
    parser.add_argument("--model-id", required=True, help="Catalogue id, e.g. distilbert-sst2")
    parser.add_argument("--out", default="exports", help="Output root; shards go to <out>/<model-id>/")
    parser.add_argument("--version", type=int, default=1, help="Catalogue version of this export")
    parser.add_argument("--cut-grid", type=int, default=1, help="Transformer layers per atom (default 1)")
    parser.add_argument("--ranges", help="Export only these atom ranges, e.g. '0-2,3-5'")
    parser.add_argument("--no-full", action="store_true", help="Do not export the full-model shard")
    parser.add_argument("--budget-ratio", type=float, default=DEFAULT_BUDGET_RATIO,
                        help="Max total size as a multiple of one model copy")
    parser.add_argument("--budget-bytes", type=int, help="Max total export size in bytes")
    parser.add_argument("--max-shard-bytes", type=int, default=DEFAULT_MAX_SHARD_BYTES)
    parser.add_argument("--phone-shard-budget-bytes", type=int, default=DEFAULT_PHONE_SHARD_BUDGET_BYTES)
    parser.add_argument("--external-data-threshold", type=int, default=DEFAULT_EXTERNAL_DATA_THRESHOLD)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--max-compositions", type=int, default=256,
                        help="Check every composition up to this many, else a covering random sample")
    parser.add_argument("--sample-text", action="append", default=[],
                        help="Extra verification text (repeatable; needs a tokenizer)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--dry-run", action="store_true", help="Print the storage report only")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing export directory")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from .exporter import export_model
        from .verify import VerificationError
    except ImportError as exc:
        print(f"crowdio-export needs the export extras: pip install 'crowdio-sdk[export]' ({exc})",
              file=sys.stderr)
        return 2
    try:
        export_model(
            args.model,
            args.model_id,
            args.out,
            cut_grid=args.cut_grid,
            ranges=parse_ranges(args.ranges) if args.ranges else None,
            include_full=not args.no_full,
            version=args.version,
            sample_texts=args.sample_text,
            tolerance=args.tolerance,
            max_compositions=args.max_compositions,
            seed=args.seed,
            budget_ratio=args.budget_ratio,
            budget_bytes=args.budget_bytes,
            max_shard_bytes=args.max_shard_bytes,
            phone_shard_budget_bytes=args.phone_shard_budget_bytes,
            external_data_threshold=args.external_data_threshold,
            opset=args.opset,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    except StorageBudgetExceeded as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except VerificationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
