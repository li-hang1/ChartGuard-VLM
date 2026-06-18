from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


QUESTION_KEYS = ("query", "question", "Question", "prompt")
ANSWER_KEYS = ("label", "answer", "answers", "Answer")
IMAGE_KEYS = ("image", "img", "picture")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a small local ChartQA manifest.")
    parser.add_argument("--dataset", default="HuggingFaceM4/ChartQA")
    parser.add_argument("--data-files", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="data/chartqa_eval")
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streaming", action="store_true")
    return parser.parse_args()


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def normalize_answer(answer: Any) -> str:
    if isinstance(answer, list):
        return str(answer[0]) if answer else ""
    if isinstance(answer, dict):
        for key in ("answer", "label", "value"):
            if key in answer:
                return normalize_answer(answer[key])
    return str(answer or "")


def save_image(image: Any, image_path: Path) -> None:
    if hasattr(image, "save"):
        image.save(image_path)
        return
    if isinstance(image, dict):
        if "path" in image and image["path"]:
            from PIL import Image

            Image.open(image["path"]).convert("RGB").save(image_path)
            return
        if "bytes" in image and image["bytes"]:
            import io
            from PIL import Image

            Image.open(io.BytesIO(image["bytes"])).convert("RGB").save(image_path)
            return
    raise ValueError(f"Unsupported image field type: {type(image)!r}")


def select_rows(dataset, limit: int, seed: int, streaming: bool):
    if streaming:
        rows = []
        for row in dataset:
            rows.append(row)
            if len(rows) >= limit:
                break
        return rows

    total = len(dataset)
    count = min(limit, total)
    rng = random.Random(seed)
    indexes = list(range(total))
    rng.shuffle(indexes)
    return [dataset[index] for index in indexes[:count]]


def main() -> None:
    args = parse_args()

    from datasets import load_dataset

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs: dict[str, Any] = {
        "split": args.split,
        "cache_dir": args.cache_dir,
        "streaming": args.streaming,
    }
    if args.data_files:
        load_kwargs["data_files"] = {args.split: args.data_files}

    dataset = load_dataset(args.dataset, **load_kwargs)
    rows = select_rows(dataset, limit=args.limit, seed=args.seed, streaming=args.streaming)

    manifest_path = output_dir / f"{args.split}_{len(rows)}.jsonl"
    kept = 0
    with manifest_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            row = dict(row)
            question = first_present(row, QUESTION_KEYS)
            answer = first_present(row, ANSWER_KEYS)
            image = first_present(row, IMAGE_KEYS)
            if question is None or answer is None or image is None:
                continue

            image_path = image_dir / f"{args.split}_{index:06d}.png"
            save_image(image, image_path)
            record = {
                "id": f"{args.split}_{index:06d}",
                "image": str(image_path),
                "question": str(question),
                "answer": normalize_answer(answer),
                "source_dataset": args.dataset,
                "split": args.split,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(json.dumps({
        "manifest": str(manifest_path),
        "kept": kept,
        "output_dir": str(output_dir),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
