"""Utilities for extracting structured JSON from model outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ParseResult:
    ok: bool
    payload: dict[str, Any] | None  # 有效载荷
    raw_json: str | None
    error: str | None = None  # 用于保存错误信息


def strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    # r表示raw，即原始字符串，^表示从头开始，()表示一个整体，?表示前面东西出现0次或一次都匹配，$表示结尾
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def extract_first_json_object(text: str) -> str | None:
    """Return the first balanced JSON object substring, if one exists."""
    text = strip_markdown_fences(text)
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False  # 当前是否在字符串内部
    escape = False  # 处理转义

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':  # 遇到 " 表示字符串结束
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def _light_repair(raw_json: str) -> str:
    repaired = raw_json.strip()
    repaired = repaired.replace("“", '"').replace("”", '"').replace("’", "'")
    # 删除json末尾}前面多余的逗号  
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)  # ()决定是否保存匹配结果供后面引用几个括号就是几个捕获组，后面\后面就跟数字几，[]是字符串集合
    return repaired


def parse_model_json(text: str) -> ParseResult:
    raw_json = extract_first_json_object(text)
    if raw_json is None:
        return ParseResult(False, None, None, "no_json_object")

    candidates = [raw_json, _light_repair(raw_json)]
    try:
        from json_repair import repair_json  # type: ignore

        # 如果导入成功，就用 repair_json 对 raw_json 做更强的 JSON 修复，然后加入候选列表。
        candidates.append(repair_json(raw_json))
    except Exception:
        pass

    last_error = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception as exc:
            last_error = str(exc)
            continue
        if not isinstance(payload, dict):  # 如果解析成功了，但解析出来的不是字典，那么返回失败
            return ParseResult(False, None, candidate, "json_root_not_object")
        return ParseResult(True, payload, candidate)

    return ParseResult(False, None, raw_json, last_error or "invalid_json")

