from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chartguard.backends import build_backend
from chartguard.pipeline import answer_with_retries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ChartGuard-VLM step-1 baseline.")
    parser.add_argument("--backend", choices=["mock", "qwen"], default="mock")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--retries", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = build_backend(
        backend=args.backend,
        model_path=args.model_path,
        metadata_path=args.metadata,
        device_map=args.device_map,
    )
    response = answer_with_retries(
        backend,
        image=args.image,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
        tolerance=args.tolerance,
        retries=args.retries,
    )
    print(json.dumps(response, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
