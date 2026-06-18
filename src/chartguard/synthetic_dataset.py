"""Synthetic chart data generation for SFT."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .prompt import build_chart_qa_prompt


@dataclass
class ChartSpec:
    chart_id: str
    chart_type: str
    title: str
    x_axis: str
    y_axis: str
    unit: str
    points: list[dict[str, Any]]


def _font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _nice_axis_max(max_value: int) -> int:
    return max(50, ((max_value + 49) // 50) * 50)


def _value_text(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _series_name(title: str) -> str:
    return title.replace(" by Quarter", "").replace(" by Month", "").replace(" by Year", "")


def make_chart_spec(rng: random.Random, chart_id: str, chart_type: str) -> ChartSpec:
    domains = [
        ("Quarterly Revenue by Quarter", "Quarter", "Revenue", "million USD", ["Q1", "Q2", "Q3", "Q4"]),
        ("Monthly Active Users by Month", "Month", "Users", "thousand users", ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]),
        ("Annual Orders by Year", "Year", "Orders", "thousand orders", ["2019", "2020", "2021", "2022", "2023"]),
        ("Customer Support Tickets by Month", "Month", "Tickets", "tickets", ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]),
        ("Product Sales by Quarter", "Quarter", "Sales", "thousand units", ["Q1", "Q2", "Q3", "Q4"]),
    ]
    title, x_axis, y_axis, unit, labels = rng.choice(domains)
    base = rng.randint(45, 180)
    points = []
    value = base
    for label in labels:
        value = max(10, value + rng.randint(-35, 45))
        points.append({"label": label, "value": int(value), "series": _series_name(title)})

    return ChartSpec(
        chart_id=chart_id,
        chart_type=chart_type,
        title=title,
        x_axis=x_axis,
        y_axis=y_axis,
        unit=unit,
        points=points,
    )


def draw_chart(spec: ChartSpec, image_path: str | Path) -> None:
    if spec.chart_type == "bar_chart":
        draw_bar_chart(spec, image_path)
        return
    if spec.chart_type == "line_chart":
        draw_line_chart(spec, image_path)
        return
    raise ValueError(f"Unsupported chart type: {spec.chart_type}")


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    spec: ChartSpec,
    width: int,
    height: int,
    max_value: int,
    margin_left: int,
    margin_right: int,
    margin_top: int,
    margin_bottom: int,
) -> None:
    title_font = _font(28)
    label_font = _font(18)
    small_font = _font(15)
    axis_color = "#303846"
    grid_color = "#d7dce2"

    draw.text((margin_left, 28), spec.title, fill="#172033", font=title_font)
    draw.text((margin_left, height - 42), spec.x_axis, fill="#172033", font=label_font)
    draw.text((18, margin_top + 120), spec.y_axis, fill="#172033", font=label_font)

    draw.line(
        [(margin_left, margin_top), (margin_left, height - margin_bottom)],
        fill=axis_color,
        width=2,
    )
    draw.line(
        [(margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom)],
        fill=axis_color,
        width=2,
    )

    plot_height = height - margin_top - margin_bottom
    tick_step = max(25, max_value // 4)
    for tick in range(0, max_value + 1, tick_step):
        y = height - margin_bottom - int(tick / max_value * plot_height)
        draw.line([(margin_left, y), (width - margin_right, y)], fill=grid_color, width=1)
        draw.text((margin_left - 62, y - 10), str(tick), fill="#394150", font=small_font)

    draw.text((width - 190, 34), f"Unit: {spec.unit}", fill="#394150", font=small_font)


def draw_bar_chart(spec: ChartSpec, image_path: str | Path) -> None:
    width, height = 900, 560
    margin_left, margin_right = 90, 50
    margin_top, margin_bottom = 80, 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = _nice_axis_max(max(int(p["value"]) for p in spec.points))

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    _draw_axes(draw, spec, width, height, max_value, margin_left, margin_right, margin_top, margin_bottom)

    label_font = _font(18)
    small_font = _font(15)
    bar_gap = max(24, 46 - len(spec.points) * 2)
    bar_width = (plot_width - bar_gap * (len(spec.points) + 1)) // len(spec.points)
    color = "#3578c8"

    for idx, point in enumerate(spec.points):
        value = int(point["value"])
        x0 = margin_left + bar_gap + idx * (bar_width + bar_gap)
        bar_height = int(value / max_value * plot_height)
        y0 = height - margin_bottom - bar_height
        x1 = x0 + bar_width
        y1 = height - margin_bottom
        draw.rectangle([x0, y0, x1, y1], fill=color)
        draw.text((x0 + bar_width // 2 - 18, y1 + 18), point["label"], fill="#172033", font=label_font)
        draw.text((x0 + bar_width // 2 - 18, y0 - 24), _value_text(value), fill="#172033", font=small_font)

    image.save(image_path)


def draw_line_chart(spec: ChartSpec, image_path: str | Path) -> None:
    width, height = 900, 560
    margin_left, margin_right = 90, 50
    margin_top, margin_bottom = 80, 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = _nice_axis_max(max(int(p["value"]) for p in spec.points))

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    _draw_axes(draw, spec, width, height, max_value, margin_left, margin_right, margin_top, margin_bottom)

    label_font = _font(18)
    small_font = _font(15)
    line_color = "#c44f2f"
    xs = []
    coords = []
    for idx, point in enumerate(spec.points):
        x = margin_left + int(idx / max(len(spec.points) - 1, 1) * plot_width)
        y = height - margin_bottom - int(int(point["value"]) / max_value * plot_height)
        xs.append(x)
        coords.append((x, y))

    draw.line(coords, fill=line_color, width=4)
    for point, (x, y) in zip(spec.points, coords):
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=line_color)
        draw.text((x - 18, height - margin_bottom + 18), point["label"], fill="#172033", font=label_font)
        draw.text((x - 18, y - 28), _value_text(point["value"]), fill="#172033", font=small_font)

    image.save(image_path)


def _payload(
    answer: str,
    chart_type: str,
    reasoning_type: str,
    evidence: list[dict[str, Any]],
    calculation: str,
    confidence: float = 1.0,
) -> dict[str, Any]:
    return {
        "answer": answer,
        "chart_type": chart_type,
        "reasoning_type": reasoning_type,
        "evidence": evidence,
        "calculation": calculation,
        "confidence": confidence,
    }


def build_questions(spec: ChartSpec) -> list[dict[str, Any]]:
    points = spec.points
    first = points[0]
    last = points[-1]
    max_point = max(points, key=lambda row: row["value"])
    min_point = min(points, key=lambda row: row["value"])
    lookup_point = points[len(points) // 2]
    difference = int(last["value"]) - int(first["value"])
    growth = (int(last["value"]) - int(first["value"])) / int(first["value"]) * 100.0

    if all(points[i]["value"] <= points[i + 1]["value"] for i in range(len(points) - 1)):
        trend_answer = "increasing"
    elif all(points[i]["value"] >= points[i + 1]["value"] for i in range(len(points) - 1)):
        trend_answer = "decreasing"
    else:
        trend_answer = "mixed"

    return [
        {
            "question": f"What is the value for {lookup_point['label']}?",
            "payload": _payload(
                answer=_value_text(lookup_point["value"]),
                chart_type=spec.chart_type,
                reasoning_type="lookup",
                evidence=[lookup_point],
                calculation="",
            ),
        },
        {
            "question": f"Which {spec.x_axis.lower()} has the highest {spec.y_axis.lower()} and what is the value?",
            "payload": _payload(
                answer=f"{max_point['label']}, {_value_text(max_point['value'])}",
                chart_type=spec.chart_type,
                reasoning_type="max",
                evidence=points,
                calculation=f"max({[p['value'] for p in points]}) = {max_point['value']} at {max_point['label']}",
            ),
        },
        {
            "question": f"Which {spec.x_axis.lower()} has the lowest {spec.y_axis.lower()} and what is the value?",
            "payload": _payload(
                answer=f"{min_point['label']}, {_value_text(min_point['value'])}",
                chart_type=spec.chart_type,
                reasoning_type="min",
                evidence=points,
                calculation=f"min({[p['value'] for p in points]}) = {min_point['value']} at {min_point['label']}",
            ),
        },
        {
            "question": f"How much did {spec.y_axis.lower()} change from {first['label']} to {last['label']}?",
            "payload": _payload(
                answer=_value_text(difference),
                chart_type=spec.chart_type,
                reasoning_type="difference",
                evidence=[first, last],
                calculation=f"{last['value']} - {first['value']} = {difference}",
            ),
        },
        {
            "question": f"What is the growth rate from {first['label']} to {last['label']}?",
            "payload": _payload(
                answer=f"{growth:.2f}%",
                chart_type=spec.chart_type,
                reasoning_type="growth_rate",
                evidence=[first, last],
                calculation=f"({last['value']} - {first['value']}) / {first['value']} * 100 = {growth:.2f}%",
            ),
        },
        {
            "question": f"What is the overall trend from {first['label']} to {last['label']}?",
            "payload": _payload(
                answer=trend_answer,
                chart_type=spec.chart_type,
                reasoning_type="trend",
                evidence=points,
                calculation="",
            ),
        },
    ]


def to_sft_record(
    chart_id: str,
    image_path: str,
    question_id: int,
    question: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    assistant_text = json.dumps(payload, ensure_ascii=False)
    return {
        "id": f"{chart_id}_q{question_id:02d}",
        "image": image_path,
        "question": question,
        "answer": payload.get("answer", ""),
        "target_json": payload,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": build_chart_qa_prompt(question)},
                ],
            },
            {"role": "assistant", "content": assistant_text},
        ],
    }

