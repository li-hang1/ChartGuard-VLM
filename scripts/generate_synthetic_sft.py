from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chartguard.synthetic_dataset import build_questions, draw_chart, make_chart_spec, to_sft_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic chart SFT data.")
    parser.add_argument("--output-dir", default="data/synthetic_sft")
    parser.add_argument("--num-charts", type=int, default=1000)
    parser.add_argument("--questions-per-chart", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chart-types", default="bar_chart,line_chart")
    return parser.parse_args()

# 把list中的dict按行写进json
def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    chart_types = [item.strip() for item in args.chart_types.split(",") if item.strip()]
    train_records: list[dict] = []
    val_records: list[dict] = []
    manifest_records: list[dict] = []

    for chart_index in range(args.num_charts):
        chart_id = f"synthetic_{chart_index:06d}"
        chart_type = rng.choice(chart_types)
        spec = make_chart_spec(rng, chart_id=chart_id, chart_type=chart_type)
        image_path = image_dir / f"{chart_id}.png"
        draw_chart(spec, image_path)

        questions = build_questions(spec)
        rng.shuffle(questions)
        selected = questions[: max(1, min(args.questions_per_chart, len(questions)))]
        is_val = rng.random() < args.val_ratio

        for question_index, item in enumerate(selected):
            record = to_sft_record(
                chart_id=chart_id,
                image_path=str(image_path),
                question_id=question_index,
                question=item["question"],
                payload=item["payload"],
            )
            if is_val:
                val_records.append(record)
            else:
                train_records.append(record)
            manifest_records.append(
                {
                    "id": record["id"],
                    "image": record["image"],
                    "question": record["question"],
                    "answer": record["answer"],
                    "chart_type": spec.chart_type,
                    "reasoning_type": item["payload"]["reasoning_type"],
                    "split": "val" if is_val else "train",
                }
            )

    all_records = train_records + val_records
    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "val.jsonl", val_records)
    write_jsonl(output_dir / "all.jsonl", all_records)
    write_jsonl(output_dir / "manifest.jsonl", manifest_records)

    summary = {
        "output_dir": str(output_dir),
        "num_charts": args.num_charts,
        "num_examples": len(all_records),
        "train_examples": len(train_records),
        "val_examples": len(val_records),
        "chart_types": chart_types,
        "questions_per_chart": args.questions_per_chart,
        "seed": args.seed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

