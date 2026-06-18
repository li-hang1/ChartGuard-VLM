from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from chartguard.backends import build_backend
from chartguard.json_utils import parse_model_json
from chartguard.verifier import verify_payload


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


def run_once(backend, image: str, question: str, max_new_tokens: int, tolerance: float):
    raw_output = backend.generate(
        image_path=image,
        question=question,
        max_new_tokens=max_new_tokens,
    )
    parsed = parse_model_json(raw_output)
    verification = None
    final = None
    if parsed.ok and parsed.payload is not None:
        verification = verify_payload(parsed.payload, tolerance=tolerance)
        if verification.corrected_payload is not None:
            final = verification.corrected_payload
        else:
            final = dict(parsed.payload)
            final["verified"] = verification.verified
            final["corrected"] = verification.corrected
            final["error_type"] = verification.error_type

    return {
        "question": question,
        "raw_output": raw_output,
        "parse": asdict(parsed),
        "verification": asdict(verification) if verification is not None else None,
        "final": final,
    }


def should_retry(attempt: dict) -> bool:
    verification = attempt.get("verification")
    if not verification:
        return True
    if verification.get("verified"):
        return False
    return verification.get("error_type") in {
        "missing_required_fields",
        "no_numeric_evidence",
        "insufficient_evidence",
        "insufficient_evidence_for_verification",
        "unsupported_or_unknown_reasoning",
        "unsupported_reasoning_type",
    }


def build_retry_question(original_question: str, attempt: dict) -> str:
    verification = attempt.get("verification") or {}
    return (
        f"{original_question}\n\n"
        "The previous answer failed deterministic verification.\n"
        f"Verifier error_type: {verification.get('error_type')}\n"
        f"Verifier errors: {verification.get('errors')}\n"
        f"Previous output: {attempt.get('raw_output')}\n\n"
        "Answer again with one valid JSON object only. If the reasoning_type is "
        "max or min, the evidence array must include every comparable chart item "
        "and its numeric value, not only the selected item. If there is not enough "
        "information, answer \"Cannot determine from the chart\"."
    )


def main() -> None:
    args = parse_args()
    backend = build_backend(
        backend=args.backend,
        model_path=args.model_path,
        metadata_path=args.metadata,
        device_map=args.device_map,
    )
    attempts = []
    attempt = run_once(
        backend,
        image=args.image,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
        tolerance=args.tolerance,
    )
    attempts.append(attempt)

    retry_budget = max(args.retries, 0)
    while retry_budget > 0 and should_retry(attempt):
        retry_budget -= 1
        retry_question = build_retry_question(args.question, attempt)
        attempt = run_once(
            backend,
            image=args.image,
            question=retry_question,
            max_new_tokens=args.max_new_tokens,
            tolerance=args.tolerance,
        )
        attempts.append(attempt)

    response = {
        "question": args.question,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "raw_output": attempts[-1]["raw_output"],
        "parse": attempts[-1]["parse"],
        "verification": attempts[-1]["verification"],
        "final": attempts[-1]["final"],
    }

    print(json.dumps(response, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
