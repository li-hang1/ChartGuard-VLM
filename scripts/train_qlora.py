from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chartguard.prompt import build_chart_qa_prompt


class SyntheticSFTDataset(Dataset):
    def __init__(self, jsonl_path: str, max_samples: int = 0) -> None:
        self.path = Path(jsonl_path)
        self.rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                self.rows.append(json.loads(line))
                if max_samples and len(self.rows) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class QwenVLCollator:
    def __init__(self, processor: Any, project_root: Path) -> None:
        self.processor = processor
        self.project_root = project_root

    def _resolve_image(self, image_path: str) -> str:
        path = Path(image_path)
        if not path.is_absolute():
            path = self.project_root / path
        return str(path.resolve())

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if len(features) != 1:
            raise ValueError("This collator expects per_device_train_batch_size=1.")

        row = features[0]
        image_path = self._resolve_image(row["image"])
        question = row["question"]
        target_json = row.get("target_json") or {}
        assistant_text = json.dumps(target_json, ensure_ascii=False)
        image = Image.open(image_path).convert("RGB")

        user_content = [
            {"type": "image", "image": image_path},
            {"type": "text", "text": build_chart_qa_prompt(question)},
        ]
        prompt_messages = [{"role": "user", "content": user_content}]
        full_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_text},
        ]

        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        inputs = self.processor(
            text=[full_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()
        prompt_len = min(prompt_inputs["input_ids"].shape[1], labels.shape[1])
        labels[:, :prompt_len] = -100
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        inputs["labels"] = labels
        return inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for ChartGuard-VLM.")
    parser.add_argument("--model-path", default="/root/LH/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--train-jsonl", default="data/synthetic_sft/train.jsonl")
    parser.add_argument("--val-jsonl", default="data/synthetic_sft/val.jsonl")
    parser.add_argument("--output-dir", default="/root/LH/models/ChartGuard-Qwen3VL-4B-QLoRA")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=64)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_device_train_batch_size != 1:
        raise ValueError("--per-device-train-batch-size must be 1 for this collator.")

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        min_pixels=56 * 56,
        max_pixels=512 * 512,
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = SyntheticSFTDataset(args.train_jsonl, max_samples=args.max_train_samples)
    eval_dataset = SyntheticSFTDataset(args.val_jsonl, max_samples=args.max_val_samples)
    collator = QwenVLCollator(processor=processor, project_root=PROJECT_ROOT)

    eval_strategy = "steps" if len(eval_dataset) > 0 and args.eval_steps > 0 else "no"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=eval_strategy,
        save_strategy="steps",
        save_total_limit=2,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    metadata = {
        "base_model": args.model_path,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "output_dir": str(output_dir),
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.target_modules,
    }
    (output_dir / "chartguard_training_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

