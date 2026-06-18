"""Deterministic verifier and correction logic for chart QA outputs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


REQUIRED_FIELDS = ("answer", "reasoning_type", "evidence")


@dataclass
class VerificationResult:
    ok: bool                                           # 整个payload是否可用
    verified: bool                                     # 答案是否经过验证
    corrected: bool = False                            # 答案是否被自动修正
    final_answer: str | None = None                    # 最终答案，可能是原答案，也可能是修正后的答案
    error_type: str | None = None                      # 错误类型，比如字段缺失、计算错误、证据不足等
    errors: list[str] = field(default_factory=list)    # 具体错误列表
    corrected_payload: dict[str, Any] | None = None    # 如果修正了答案，这里保存修正后的完整 payload

# 从任意输入中提取数字，并转成 float。
def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)

    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)  # 找出字符串中出现的第一个数字
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None

# 从答案中提取数字
def extract_answer_number(answer: Any) -> float | None:
    return parse_number(answer)

# 将推理类型标准化
def normalize_reasoning_type(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    aliases = {
        "maximum": "max",
        "highest": "max",
        "minimum": "min",
        "lowest": "min",
        "subtract": "difference",
        "diff": "difference",
        "percentage_change": "growth_rate",
        "growth": "growth_rate",
        "rate": "growth_rate",
    }
    return aliases.get(text, text)

# 把 evidence 统一转成 list[dict] 格式。
# 每个字典是{"label": "", "value": ""}的形式 
def normalize_evidence(evidence: Any) -> list[dict[str, Any]]:
    if isinstance(evidence, dict):
        rows = []
        for key, value in evidence.items():
            if isinstance(value, dict):
                row = {"label": key}
                row.update(value)
            else:
                row = {"label": key, "value": value}
            rows.append(row)
        return rows

    if isinstance(evidence, list):
        rows = []
        for index, item in enumerate(evidence):
            if isinstance(item, dict):
                rows.append(item)
            else:
                rows.append({"label": str(index), "value": item})
        return rows

    return []

# 从 payload 的 evidence 中提取可计算的数据点。
# 变成[(label, value)]的形式，value是纯数值
def numeric_points(payload: dict[str, Any]) -> list[tuple[str, float]]:
    points = []
    for row in normalize_evidence(payload.get("evidence")):
        label = str(row.get("label") or row.get("x") or row.get("name") or "")
        value = parse_number(row.get("value") or row.get("y"))
        if label and value is not None:
            points.append((label, value))
    return points

# 判断答案中是否包含某个标签。忽略大小写
def answer_contains_label(answer: Any, label: str) -> bool:
    return label.lower() in str(answer or "").lower()

# 判断答案里的数字是否接近预期值
def answer_number_close(answer: Any, expected: float, tolerance: float) -> bool:
    observed = extract_answer_number(answer)
    if observed is None:
        return False
    scale = max(abs(expected), 1.0)
    return abs(observed - expected) / scale <= tolerance

# 构造一个“已修正”的验证结果。
def with_correction(
    payload: dict[str, Any],
    final_answer: str,
    error_type: str,
) -> VerificationResult:
    corrected = dict(payload)
    corrected["original_answer"] = payload.get("answer")
    corrected["answer"] = final_answer
    corrected["verified"] = True
    corrected["corrected"] = True
    corrected["error_type"] = error_type
    return VerificationResult(
        ok=True,
        verified=True,
        corrected=True,
        final_answer=final_answer,
        error_type=error_type,
        corrected_payload=corrected,
    )


def verify_payload(payload: dict[str, Any], tolerance: float = 0.02) -> VerificationResult:
    errors = [field for field in REQUIRED_FIELDS if field not in payload]
    if errors:
        return VerificationResult(
            ok=False,
            verified=False,
            error_type="missing_required_fields",
            errors=errors,
        )

    answer = payload.get("answer")
    reasoning_type = normalize_reasoning_type(payload.get("reasoning_type"))
    points = numeric_points(payload)

    if reasoning_type in {"unknown", ""}:
        return VerificationResult(
            ok=False,
            verified=False,
            final_answer=str(answer),
            error_type="unsupported_or_unknown_reasoning",
        )

    # 对于lookup, trend, comparison这三个类型，只检查evidence是否为空
    if reasoning_type in {"lookup", "trend", "comparison"}:
        if normalize_evidence(payload.get("evidence")):
            return VerificationResult(ok=True, verified=True, final_answer=str(answer))
        return VerificationResult(
            ok=False,
            verified=False,
            final_answer=str(answer),
            error_type="empty_evidence",
        )

    # 对于max、min、difference、growth_rate这些类型，需要数值证据
    if len(points) == 0:
        return VerificationResult(
            ok=False,
            verified=False,
            final_answer=str(answer),
            error_type="no_numeric_evidence",
        )

    if reasoning_type == "max":
        if len(points) < 2:
            return VerificationResult(
                ok=False,
                verified=False,
                final_answer=str(answer),
                error_type="insufficient_evidence_for_verification",
                errors=["max requires at least two numeric evidence points"],
            )
        label, value = max(points, key=lambda item: item[1])
        if answer_contains_label(answer, label) or answer_number_close(answer, value, tolerance):
            return VerificationResult(ok=True, verified=True, final_answer=str(answer))
        return with_correction(payload, f"{label}, {value:g}", "answer_evidence_mismatch")

    if reasoning_type == "min":
        if len(points) < 2:
            return VerificationResult(
                ok=False,
                verified=False,
                final_answer=str(answer),
                error_type="insufficient_evidence_for_verification",
                errors=["min requires at least two numeric evidence points"],
            )
        label, value = min(points, key=lambda item: item[1])
        if answer_contains_label(answer, label) or answer_number_close(answer, value, tolerance):
            return VerificationResult(ok=True, verified=True, final_answer=str(answer))
        return with_correction(payload, f"{label}, {value:g}", "answer_evidence_mismatch")

    if reasoning_type == "difference":
        if len(points) < 2:
            return VerificationResult(
                ok=False,
                verified=False,
                final_answer=str(answer),
                error_type="insufficient_evidence",
            )
        first_label, first_value = points[0]
        second_label, second_value = points[1]
        expected = second_value - first_value
        if answer_number_close(answer, expected, tolerance):
            return VerificationResult(ok=True, verified=True, final_answer=str(answer))
        final = f"{second_label} - {first_label} = {expected:g}"
        return with_correction(payload, final, "calculation_error")

    if reasoning_type == "growth_rate":
        if len(points) < 2:
            return VerificationResult(
                ok=False,
                verified=False,
                final_answer=str(answer),
                error_type="insufficient_evidence",
            )
        first_label, first_value = points[0]
        second_label, second_value = points[1]
        if first_value == 0:
            return VerificationResult(
                ok=False,
                verified=False,
                final_answer=str(answer),
                error_type="division_by_zero",
            )
        expected = (second_value - first_value) / first_value * 100.0
        if answer_number_close(answer, expected, tolerance):
            return VerificationResult(ok=True, verified=True, final_answer=str(answer))
        final = f"{second_label} vs {first_label}: {expected:.2f}%"
        return with_correction(payload, final, "calculation_error")

    return VerificationResult(
        ok=False,
        verified=False,
        final_answer=str(answer),
        error_type="unsupported_reasoning_type",
        errors=[reasoning_type],
    )
