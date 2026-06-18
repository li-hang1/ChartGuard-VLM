from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chartguard.backends import build_backend
from chartguard.metrics import summarize_results
from chartguard.pipeline import answer_with_retries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ChartGuard-VLM on a ChartQA manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--backend", choices=["mock", "qwen"], default="qwen")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--output", default="outputs/chartqa_eval/results.jsonl")
    parser.add_argument("--summary-output", default="outputs/chartqa_eval/summary.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--numeric-tolerance", type=float, default=0.05)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def iter_manifest(path: str, limit: int):
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)
            count += 1
            if limit and count >= limit:
                return


def main() -> None:
    args = parse_args()
    backend = build_backend(
        backend=args.backend,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        device_map=args.device_map,
    )

    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    started = time.time()
    with output_path.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(iter_manifest(args.manifest, args.limit), start=1):
            result = answer_with_retries(
                backend=backend,
                image=sample["image"],
                question=sample["question"],
                max_new_tokens=args.max_new_tokens,
                tolerance=args.tolerance,
                retries=args.retries,
            )
            row = {
                "id": sample.get("id") or str(index),
                "image": sample["image"],
                "question": sample["question"],
                "reference_answer": sample.get("answer", ""),
                **result,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            results.append(row)

            final = row.get("final") or {}
            answer = final.get("answer") if isinstance(final, dict) else ""
            attempts = row.get("attempts") or []
            first_parse = attempts[0].get("parse") if attempts else {}
            first_payload = (first_parse or {}).get("payload") or {}
            raw_answer = first_payload.get("answer") if isinstance(first_payload, dict) else ""
            last_parse = attempts[-1].get("parse") if attempts else {}
            last_payload = (last_parse or {}).get("payload") or {}
            retry_answer = last_payload.get("answer") if isinstance(last_payload, dict) else ""
            verified = final.get("verified") if isinstance(final, dict) else False
            print(
                json.dumps(
                    {
                        "index": index,
                        "id": row["id"],
                        "raw_pred": raw_answer,
                        "retry_pred": retry_answer,
                        "final_pred": answer,
                        "ref": row["reference_answer"],
                        "verified": verified,
                    },
                    ensure_ascii=False,
                )
            )

    summary = summarize_results(results, numeric_tolerance=args.numeric_tolerance)
    summary["seconds"] = round(time.time() - started, 2)
    summary["manifest"] = args.manifest
    summary["output"] = str(output_path)
    summary["model_path"] = args.model_path
    summary["adapter_path"] = args.adapter_path
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
