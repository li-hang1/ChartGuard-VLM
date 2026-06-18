from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chartguard.backends import build_backend
from chartguard.json_utils import parse_model_json
from chartguard.verifier import verify_payload


def make_app(model_path: str, backend_name: str = "qwen") -> gr.Blocks:
    backend = build_backend(backend=backend_name, model_path=model_path)

    def ask(image, question):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            image.save(handle.name)
            image_path = handle.name

        raw = backend.generate(image_path=image_path, question=question)
        parsed = parse_model_json(raw)
        if not parsed.ok or parsed.payload is None:
            return raw, json.dumps(asdict(parsed), indent=2, ensure_ascii=False), "parse_failed"

        verification = verify_payload(parsed.payload)
        final_payload = verification.corrected_payload or dict(parsed.payload)
        final_payload["verified"] = verification.verified
        final_payload["corrected"] = verification.corrected
        final_payload["error_type"] = verification.error_type
        Path(image_path).unlink(missing_ok=True)
        return (
            raw,
            json.dumps(final_payload, indent=2, ensure_ascii=False),
            verification.final_answer or str(final_payload.get("answer")),
        )

    with gr.Blocks(title="ChartGuard-VLM") as demo:
        gr.Markdown("# ChartGuard-VLM")
        with gr.Row():
            image = gr.Image(type="pil", label="Chart image")
            question = gr.Textbox(label="Question")
        run = gr.Button("Ask")
        answer = gr.Textbox(label="Final answer")
        final_json = gr.Code(label="Verified JSON", language="json")
        raw_output = gr.Textbox(label="Raw model output")
        run.click(ask, inputs=[image, question], outputs=[raw_output, final_json, answer])
    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["qwen", "mock"], default="qwen")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = make_app(model_path=args.model_path, backend_name=args.backend)
    demo.launch(server_name=args.host, server_port=args.port)
