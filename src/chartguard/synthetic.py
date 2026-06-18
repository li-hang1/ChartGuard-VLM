"""Generate small synthetic chart samples with Pillow."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


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


def generate_revenue_bar_chart(output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_path = output / "revenue_bar.png"
    meta_path = output / "revenue_bar.meta.json"

    points = [
        {"label": "Q1", "value": 120},
        {"label": "Q2", "value": 150},
        {"label": "Q3", "value": 130},
        {"label": "Q4", "value": 180},
    ]

    width, height = 900, 560
    margin_left, margin_right = 90, 50
    margin_top, margin_bottom = 80, 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = 200

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(28)
    label_font = _font(20)
    small_font = _font(16)

    draw.text((margin_left, 28), "Quarterly Revenue", fill="#172033", font=title_font)
    draw.text((margin_left, height - 42), "Quarter", fill="#172033", font=label_font)
    draw.text((18, margin_top + 120), "Revenue", fill="#172033", font=label_font)

    axis_color = "#303846"
    grid_color = "#d7dce2"
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

    for tick in range(0, max_value + 1, 50):
        y = height - margin_bottom - int(tick / max_value * plot_height)
        draw.line([(margin_left, y), (width - margin_right, y)], fill=grid_color, width=1)
        draw.text((margin_left - 55, y - 10), str(tick), fill="#394150", font=small_font)

    bar_gap = 46
    bar_width = (plot_width - bar_gap * (len(points) + 1)) // len(points)
    color = "#3578c8"
    for idx, point in enumerate(points):
        x0 = margin_left + bar_gap + idx * (bar_width + bar_gap)
        bar_height = int(point["value"] / max_value * plot_height)
        y0 = height - margin_bottom - bar_height
        x1 = x0 + bar_width
        y1 = height - margin_bottom
        draw.rectangle([x0, y0, x1, y1], fill=color)
        draw.text((x0 + bar_width // 2 - 16, y1 + 18), point["label"], fill="#172033", font=label_font)
        draw.text((x0 + bar_width // 2 - 18, y0 - 26), str(point["value"]), fill="#172033", font=small_font)

    image.save(image_path)
    meta = {
        "title": "Quarterly Revenue",
        "unit": "million USD",
        "points": points,
        "sample_questions": [
            {
                "question": "Which quarter has the highest revenue and how much is it?",
                "answer": "Q4, 180",
                "reasoning_type": "max",
            },
            {
                "question": "What is the growth rate from Q1 to Q4?",
                "answer": "50.00%",
                "reasoning_type": "growth_rate",
            },
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return image_path, meta_path

