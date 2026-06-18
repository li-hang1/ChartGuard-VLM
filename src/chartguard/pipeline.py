"""Reusable inference pipeline with verifier-driven retry."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .json_utils import parse_model_json
from .verifier import verify_payload


def run_once(
    backend: Any,
    image: str,
    question: str,
    max_new_tokens: int,
    tolerance: float,
) -> dict[str, Any]:
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


def should_retry(attempt: dict[str, Any]) -> bool:
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


def build_retry_question(original_question: str, attempt: dict[str, Any]) -> str:
    verification = attempt.get("verification") or {}
    return (
        f"{original_question}\n\n"
        "The previous answer failed deterministic verification.\n"
        f"Verifier error_type: {verification.get('error_type')}\n"
        f"Verifier errors: {verification.get('errors')}\n"
        f"Previous output: {attempt.get('raw_output')}\n\n"
        "Answer again with one valid JSON object only. If the reasoning_type is "
        "max or min, the evidence array must include every comparable chart item "
        "and its numeric value, not only the selected item. If the reasoning_type "
        "is difference or growth_rate, evidence must include the two mentioned "
        "items and their numeric values. For growth_rate, use "
        "(target_value - base_value) / base_value * 100. If the question says "
        "\"from A to B\", A is the base and B is the target. If there is not "
        "enough information, answer \"Cannot determine from the chart\"."
    )


def answer_with_retries(
    backend: Any,
    image: str,
    question: str,
    max_new_tokens: int = 512,
    tolerance: float = 0.02,
    retries: int = 1,
) -> dict[str, Any]:
    attempts = []
    attempt = run_once(
        backend,
        image=image,
        question=question,
        max_new_tokens=max_new_tokens,
        tolerance=tolerance,
    )
    attempts.append(attempt)

    retry_budget = max(retries, 0)
    while retry_budget > 0 and should_retry(attempt):
        retry_budget -= 1
        retry_question = build_retry_question(question, attempt)
        attempt = run_once(
            backend,
            image=image,
            question=retry_question,
            max_new_tokens=max_new_tokens,
            tolerance=tolerance,
        )
        attempts.append(attempt)

    return {
        "question": question,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "raw_output": attempts[-1]["raw_output"],
        "parse": attempts[-1]["parse"],
        "verification": attempts[-1]["verification"],
        "final": attempts[-1]["final"],
    }

