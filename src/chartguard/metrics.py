"""Metrics for chart QA evaluation."""

from __future__ import annotations

import re
from typing import Any

from .verifier import parse_number


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,:;!?")
    return text


def extract_reference_answer(answer: Any) -> str:
    if isinstance(answer, list):
        if len(answer) == 0:
            return ""
        return str(answer[0])
    if isinstance(answer, dict):
        for key in ("answer", "label", "value"):
            if key in answer:
                return str(answer[key])
        return str(answer)
    return str(answer or "")


def final_answer_text(result: dict[str, Any]) -> str:
    final = result.get("final") or {}
    if isinstance(final, dict):
        return str(final.get("answer") or "")
    return ""


def attempt_answer_text(result: dict[str, Any], index: int) -> str:
    attempts = result.get("attempts") or []
    if not attempts:
        return ""
    try:
        attempt = attempts[index]
    except IndexError:
        return ""

    parse = attempt.get("parse") or {}
    payload = parse.get("payload") or {}
    if isinstance(payload, dict):
        return str(payload.get("answer") or "")
    return ""


def relaxed_match(prediction: Any, reference: Any, numeric_tolerance: float = 0.05) -> bool:
    pred_text = normalize_text(prediction)
    ref_text = normalize_text(reference)
    if not pred_text or not ref_text:
        return False
    if pred_text == ref_text:
        return True
    if pred_text in ref_text or ref_text in pred_text:
        return True

    pred_num = parse_number(prediction)
    ref_num = parse_number(reference)
    if pred_num is None or ref_num is None:
        return False

    scale = max(abs(ref_num), 1.0)
    return abs(pred_num - ref_num) / scale <= numeric_tolerance


def exact_match(prediction: Any, reference: Any) -> bool:
    return normalize_text(prediction) == normalize_text(reference)


def summarize_results(results: list[dict[str, Any]], numeric_tolerance: float = 0.05) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "raw_exact_match": 0.0,
            "raw_relaxed_accuracy": 0.0,
            "retry_exact_match": 0.0,
            "retry_relaxed_accuracy": 0.0,
            "final_exact_match": 0.0,
            "final_relaxed_accuracy": 0.0,
            "verifier_retry_exact_gain": 0.0,
            "verifier_retry_relaxed_gain": 0.0,
            "exact_match": 0.0,
            "relaxed_accuracy": 0.0,
            "json_valid_rate": 0.0,
            "verified_rate": 0.0,
            "corrected_rate": 0.0,
            "avg_attempts": 0.0,
        }

    exact = 0
    relaxed = 0
    raw_exact = 0
    raw_relaxed = 0
    retry_exact = 0
    retry_relaxed = 0
    json_valid = 0
    verified = 0
    corrected = 0
    attempts = 0

    for row in results:
        reference = row.get("reference_answer")
        prediction = final_answer_text(row)
        raw_prediction = attempt_answer_text(row, 0)
        retry_prediction = attempt_answer_text(row, -1)
        exact += int(exact_match(prediction, reference))
        relaxed += int(relaxed_match(prediction, reference, numeric_tolerance=numeric_tolerance))
        raw_exact += int(exact_match(raw_prediction, reference))
        raw_relaxed += int(relaxed_match(raw_prediction, reference, numeric_tolerance=numeric_tolerance))
        retry_exact += int(exact_match(retry_prediction, reference))
        retry_relaxed += int(relaxed_match(retry_prediction, reference, numeric_tolerance=numeric_tolerance))
        parse = row.get("parse") or {}
        json_valid += int(bool(parse.get("ok")))
        final = row.get("final") or {}
        if isinstance(final, dict):
            verified += int(bool(final.get("verified")))
            corrected += int(bool(final.get("corrected")))
        attempts += int(row.get("attempt_count") or 0)

    return {
        "total": total,
        "raw_exact_match": raw_exact / total,
        "raw_relaxed_accuracy": raw_relaxed / total,
        "retry_exact_match": retry_exact / total,
        "retry_relaxed_accuracy": retry_relaxed / total,
        "final_exact_match": exact / total,
        "final_relaxed_accuracy": relaxed / total,
        "verifier_retry_exact_gain": (exact - raw_exact) / total,
        "verifier_retry_relaxed_gain": (relaxed - raw_relaxed) / total,
        "exact_match": exact / total,
        "relaxed_accuracy": relaxed / total,
        "json_valid_rate": json_valid / total,
        "verified_rate": verified / total,
        "corrected_rate": corrected / total,
        "avg_attempts": attempts / total,
    }
