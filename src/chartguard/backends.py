"""Inference backends for ChartGuard-VLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .prompt import build_chart_qa_prompt


class MockChartBackend:
    """Backend used for smoke tests before model weights are downloaded."""

    def __init__(self, metadata_path: str | None = None) -> None:
        self.metadata = {}
        if metadata_path:
            self.metadata = json.loads(Path(metadata_path).read_text())

    def generate(self, image_path: str, question: str, max_new_tokens: int = 512) -> str:
        points = self.metadata.get("points") or [
            {"label": "Q1", "value": 120},
            {"label": "Q2", "value": 150},
            {"label": "Q3", "value": 130},
            {"label": "Q4", "value": 180},
        ]
        lowered = question.lower()
        if any(word in lowered for word in ["highest", "max", "largest", "最高"]):
            best = max(points, key=lambda item: float(item["value"]))
            payload = {
                "answer": f"{best['label']}, {best['value']}",
                "chart_type": "bar_chart",
                "reasoning_type": "max",
                "evidence": points,
                "calculation": "max(evidence)",
                "confidence": 1.0,
            }
        elif any(word in lowered for word in ["growth", "increase", "增长"]):
            first = points[0]
            last = points[-1]
            rate = (float(last["value"]) - float(first["value"])) / float(first["value"]) * 100
            payload = {
                "answer": f"{rate:.2f}%",
                "chart_type": "bar_chart",
                "reasoning_type": "growth_rate",
                "evidence": [first, last],
                "calculation": f"({last['value']} - {first['value']}) / {first['value']} * 100",
                "confidence": 1.0,
            }
        else:
            payload = {
                "answer": "Cannot determine from the chart",
                "chart_type": "bar_chart",
                "reasoning_type": "unknown",
                "evidence": [],
                "calculation": "",
                "confidence": 0.0,
            }
        return json.dumps(payload, ensure_ascii=False)


class QwenVLBackend:
    """Qwen2.5-VL / Qwen3-VL inference backend using Hugging Face Transformers."""

    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        dtype: str = "auto",
        device_map: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.adapter_path = adapter_path

        import torch
        from transformers import AutoProcessor

        self.torch = torch
        processor_path = adapter_path or model_path
        self.processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)  # 返回Processor对象
        self.model = self._load_model(model_path, dtype=dtype, device_map=device_map)
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def _input_device(self) -> Any:
        device = getattr(self.model, "device", None)
        if device is not None:
            return device
        return next(self.model.parameters()).device

    def _load_model(self, model_path: str, dtype: str, device_map: str) -> Any:
        """
        model_path: 模型路径，可以是本地路径，也可以是 Hugging Face 模型名。
        bdtype: 模型参数的数据类型。
        device_map: 模型放到哪个设备。
        return: 返回模型对象。
        """
        from transformers import AutoModelForCausalLM  # 导入普通因果语言模型加载类。

        # 构造一个字典，保存加载模型时要传入的参数。
        # 类型是auto的时候，优先读取模型配置文件中的推荐类型
        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": True,
        }

        # 定义一个候选模型类列表，程序会按顺序尝试这些类。
        candidates = [
            ("transformers", "Qwen3VLForConditionalGeneration"),
            ("transformers", "Qwen2_5_VLForConditionalGeneration"),
            ("transformers", "Qwen2VLForConditionalGeneration"),
            ("transformers", "AutoModelForImageTextToText"),
        ]
        for module_name, class_name in candidates:
            try:
                module = __import__(module_name, fromlist=[class_name])
                model_cls = getattr(module, class_name)
            except Exception:
                continue
            try:
                return model_cls.from_pretrained(model_path, **model_kwargs)
            except Exception:
                continue

        return AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    def generate(self, image_path: str, question: str, max_new_tokens: int = 512) -> str:
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as exc:
            raise RuntimeError(
                "qwen-vl-utils is required for Qwen-VL inference. "
                "Install it with: pip install -U qwen-vl-utils"
            ) from exc

        prompt = build_chart_qa_prompt(question)
        image_uri = Path(image_path).resolve().as_uri()  # .as_uri()转化成统一资源标识符 (Uniform Resource Identifier)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # tokenize=False时apply_chat_template返回字符串
        # processor.apply_chat_template作用是把message处理成标准格式
        # add_generation_prompt作用是在对话末尾自动添加一个“助手开始回答”的提示标记（generation prompt）
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # process_vision_info用来提取所有视觉内容（图片、视频），并转换成 processor 可以直接处理的格式。
        # image和video如果哪个没有的话，哪个就是None
        image_inputs, video_inputs = process_vision_info(messages)
        # return_tensors="pt"表示输出tensor，如果指定np就是返回numpy数组
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # inputs是<class 'transformers.feature_extraction_utils.BatchFeature'>对象，类似一个字典
        # 有input_ids，attention_mask，pixel_values，image_grid_thw这四个键，值都是tensor
        inputs = inputs.to(self._input_device())

        with self.torch.no_grad():
            # do_sample决定按概率采样还是每次选概率最大的那个
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # self.processor.batch_decode返回字符串列表
        # clean_up_tokenization_spaces作用是解码后是否自动清理 tokenizer 产生的一些多余空格和格式。
        output = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output[0]


def build_backend(
    backend: str,
    model_path: str | None = None,
    adapter_path: str | None = None,
    metadata_path: str | None = None,
    device_map: str = "auto",
) -> MockChartBackend | QwenVLBackend:
    if backend == "mock":
        return MockChartBackend(metadata_path)
    if backend == "qwen":
        if not model_path:
            raise ValueError("--model-path is required when --backend qwen")
        return QwenVLBackend(model_path=model_path, adapter_path=adapter_path, device_map=device_map)
    raise ValueError(f"Unsupported backend: {backend}")
