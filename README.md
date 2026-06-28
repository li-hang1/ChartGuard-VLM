# ChartGuard-VLM

ChartGuard-VLM 是一个面向图表问答的可验证多模态大模型项目。系统输入一张图表图片和一个自然语言问题，调用多模态大模型生成结构化 JSON，再用确定性 verifier 检查证据、计算过程和最终答案，必要时触发 retry 让模型重新回答。

这个项目的重点不是只做一次图表问答，而是建立完整闭环：

- 多模态模型推理
- 结构化 JSON 输出
- 证据与计算过程校验
- verifier-driven retry
- ChartQA 批量评测
- 合成图表 SFT 数据生成
- QLoRA 单卡微调
- 微调前后效果对比

## 项目功能

当前系统已经实现：

- 图表图片 + 问题输入，输出结构化 JSON。
- Qwen-VL / Qwen3-VL 推理 backend。
- mock backend，用于不加载模型时做 smoke test。
- 模型输出 JSON 抽取和轻量修复。
- verifier 校验常见图表推理类型：
  - lookup
  - max / min
  - difference
  - growth_rate
  - trend
- verifier 失败时自动构造纠错 prompt 并 retry。
- ChartQA 测试集批量评测。
- raw / retry / final 三组准确率统计。
- 合成图表 SFT 数据生成。
- QLoRA 微调 Qwen3-VL-4B。
- 加载 base model + LoRA adapter 做微调后评测。

## 环境准备

核心依赖包括：

```bash
pip install -U "transformers>=4.57.0" qwen-vl-utils json-repair peft
```

如果要运行 Web demo，可以额外安装：

```bash
pip install -U gradio
```

## 命令行下载模型

单张 RTX 3090 上建议先使用较小模型验证流程：

```bash
hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir /root/code/models/Qwen2.5-VL-3B-Instruct
```

如果磁盘空间和 Transformers 版本满足要求，可以使用 Qwen3-VL-4B：

```bash
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /root/code/models/Qwen3-VL-4B-Instruct
```

## 不加载模型的 smoke test

先生成一张合成图表：

```bash
python scripts/make_synthetic_sample.py
```

使用 mock backend 跑完整 parser + verifier 流程：

```bash
python scripts/run_baseline.py \
  --backend mock \
  --image examples/revenue_bar.png \
  --question "Which quarter has the highest revenue and how much is it?" \
  --metadata examples/revenue_bar.meta.json
```

## 运行真实 VLM 推理

```bash
python scripts/run_baseline.py \
  --backend qwen \
  --model-path ./models/Qwen3-VL-4B-Instruct \
  --image examples/revenue_bar.png \
  --question "Which quarter has the highest revenue and how much is it?" \
  --max-new-tokens 512 \
  --retries 1
```

脚本会输出：

- 原始模型输出
- JSON 解析结果
- verifier 校验结果
- 修正或 fallback 后的最终答案
- retry 尝试记录

如果遇到 `ModuleNotFoundError: No module named 'chartguard'`，优先确认命令是在项目根目录运行，并设置了：

```bash
export PYTHONPATH=$PWD/src
```

## 模型输出格式

模型被要求只输出一个 JSON 对象，例如：

```json
{
  "answer": "Q4, 180",
  "chart_type": "bar_chart",
  "reasoning_type": "max",
  "evidence": [
    {"label": "Q1", "value": 120},
    {"label": "Q2", "value": 150},
    {"label": "Q3", "value": 130},
    {"label": "Q4", "value": 180}
  ],
  "calculation": "max([120, 150, 130, 180]) = 180 at Q4",
  "confidence": 0.85
}
```

字段含义：

- `answer`：最终答案。
- `chart_type`：图表类型。
- `reasoning_type`：推理类型，例如 lookup、max、difference、growth_rate。
- `evidence`：支持答案的图表证据。
- `calculation`：计算过程。
- `confidence`：模型置信度。

## verifier 逻辑

verifier 不直接相信模型答案，而是根据 `reasoning_type` 和 `evidence` 重新计算或检查答案。

它会返回以下几类结果：

- 通过校验的答案。
- 可被规则修正的答案。
- 低置信 fallback。
- 不支持或不可验证的拒答。

如果 verifier 发现可恢复错误，例如缺少数值证据、证据不完整、推理类型无法校验，pipeline 会构造包含错误原因的 retry prompt，让模型重新输出合法 JSON。

## 准备 ChartQA 测试集

ChartQA test 作为最终 benchmark，不用于训练。

如果 Hugging Face 访问稳定，可以直接 streaming：

```bash
HF_ENDPOINT=https://hf-mirror.com python scripts/prepare_chartqa.py \
  --dataset HuggingFaceM4/ChartQA \
  --split test \
  --output-dir data/chartqa_eval \
  --cache-dir data/hf_cache \
  --limit 32 \
  --streaming
```

如果 streaming 不稳定，先命令行下载 test parquet：

```bash
HF_HUB_DISABLE_XET=1 HF_ENDPOINT=https://hf-mirror.com hf download \
  HuggingFaceM4/ChartQA \
  --repo-type dataset \
  --local-dir data/chartqa_raw \
  --include 'data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet' \
  --max-workers 1
```

然后从本地 parquet 生成 manifest：

```bash
python scripts/prepare_chartqa.py \
  --dataset parquet \
  --data-files data/chartqa_raw/data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet \
  --split test \
  --output-dir data/chartqa_eval \
  --cache-dir data/hf_cache \
  --limit 2500
```

生成结果包括：

```text
data/chartqa_eval/test_2500.jsonl
data/chartqa_eval/images/test_000000.png
```

如果直接从 Hugging Face 数据集读取，`--dataset` 填：

```text
HuggingFaceM4/ChartQA
```

如果已经下载了 parquet 文件，`--dataset` 填：

```text
parquet
```

## 评测原始模型

小样本评测：

```bash
python scripts/evaluate_chartqa.py \
  --manifest data/chartqa_eval/test_32.jsonl \
  --backend qwen \
  --model-path ./models/Qwen3-VL-4B-Instruct \
  --output outputs/chartqa_eval/qwen3vl_4b_test32.jsonl \
  --summary-output outputs/chartqa_eval/qwen3vl_4b_test32_summary.json \
  --limit 32 \
  --retries 1
```

完整 2500 条测试集评测：

```bash
python scripts/evaluate_chartqa.py \
  --manifest data/chartqa_eval/test_2500.jsonl \
  --backend qwen \
  --model-path ./models/Qwen3-VL-4B-Instruct \
  --output outputs/chartqa_eval/qwen3vl_2500.jsonl \
  --summary-output outputs/chartqa_eval/qwen3vl_2500_summary.json \
  --retries 1
```

summary 会包含：

- `raw_exact_match`
- `raw_relaxed_accuracy`
- `retry_exact_match`
- `retry_relaxed_accuracy`
- `final_exact_match`
- `final_relaxed_accuracy`
- `verifier_retry_exact_gain`
- `verifier_retry_relaxed_gain`
- `json_valid_rate`
- `verified_rate`
- `corrected_rate`
- `avg_attempts`

其中：

- `raw_*`：第一次模型输出的答案准确率。
- `retry_*`：如果触发 retry，最后一次模型输出的答案准确率。
- `final_*`：经过 verifier 修正或 fallback 后的最终答案准确率。
- `json_valid_rate`：模型输出能被解析为合法 JSON 的比例。
- `verified_rate`：最终答案通过 verifier 的比例。
- `corrected_rate`：verifier 对答案做出规则修正的比例。

## 生成合成 SFT 数据

ChartQA test 只做测试集。SFT 数据由 Python 自动生成合成图表和标准答案，包含完整的 `answer`、`reasoning_type`、`evidence` 和 `calculation` 字段。

生成小规模 smoke 数据：

```bash
python scripts/generate_synthetic_sft.py \
  --output-dir data/synthetic_sft_smoke \
  --num-charts 20 \
  --questions-per-chart 5 \
  --val-ratio 0.1 \
  --seed 42
```

生成正式 SFT 数据：

```bash
python scripts/generate_synthetic_sft.py \
  --output-dir data/synthetic_sft \
  --num-charts 1000 \
  --questions-per-chart 5 \
  --val-ratio 0.1 \
  --seed 42
```

输出文件：

```text
data/synthetic_sft/images/*.png
data/synthetic_sft/train.jsonl
data/synthetic_sft/val.jsonl
data/synthetic_sft/all.jsonl
data/synthetic_sft/manifest.jsonl
data/synthetic_sft/summary.json
```

每条 SFT 样本包含：

- 图表图片路径
- 问题
- 目标 JSON
- Qwen 风格多模态 messages

## QLoRA 微调

先保留一份原始模型：

```bash
cp -al ./models/Qwen3-VL-4B-Instruct ./models/Qwen3-VL-4B-Instruct-original
```

这里使用 hardlink 方式备份，节省磁盘空间。训练只读这个目录，不会覆盖原始模型权重。

先跑 1 step smoke test：

```bash
python scripts/train_qlora.py \
  --model-path ./models/Qwen3-VL-4B-Instruct-original \
  --train-jsonl data/synthetic_sft/train.jsonl \
  --val-jsonl data/synthetic_sft/val.jsonl \
  --output-dir ./models/ChartGuard-Qwen3VL-4B-QLoRA-smoke \
  --max-train-samples 2 \
  --max-val-samples 1 \
  --max-steps 1 \
  --save-steps 1 \
  --eval-steps 0
```

正式 QLoRA 微调：

```bash
python scripts/train_qlora.py \
  --model-path ./models/Qwen3-VL-4B-Instruct-original \
  --train-jsonl data/synthetic_sft/train.jsonl \
  --val-jsonl data/synthetic_sft/val.jsonl \
  --output-dir ./models/ChartGuard-Qwen3VL-4B-QLoRA \
  --num-train-epochs 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4
```

训练输出目录保存的是 LoRA adapter 和 processor/tokenizer 文件，不是合并后的完整模型：

```text
./models/ChartGuard-Qwen3VL-4B-QLoRA/adapter_model.safetensors
./models/ChartGuard-Qwen3VL-4B-QLoRA/adapter_config.json
./models/ChartGuard-Qwen3VL-4B-QLoRA/chartguard_training_config.json
```

QLoRA 训练目标仍然是 causal language modeling：给定图像 token、用户问题 token 和前文回答 token，预测 assistant JSON 的下一个 token。训练时 prompt 部分 label 被 mask，只对 assistant 的目标 JSON 计算 loss。

## 评测微调后的模型

微调后推理需要加载 base model + LoRA adapter：

```bash
python scripts/evaluate_chartqa.py \
  --manifest data/chartqa_eval/test_2500.jsonl \
  --backend qwen \
  --model-path ./models/Qwen3-VL-4B-Instruct-original \
  --adapter-path ./models/ChartGuard-Qwen3VL-4B-QLoRA \
  --output outputs/chartqa_eval/qwen3vl_qlora_2500.jsonl \
  --summary-output outputs/chartqa_eval/qwen3vl_qlora_2500_summary.json \
  --retries 1 \
  --max-new-tokens 512
```

建议对比以下文件：

```text
outputs/chartqa_eval/qwen3vl_2500_summary.json
outputs/chartqa_eval/qwen3vl_qlora_2500_summary.json
```

重点看三类指标：

- 答案准确率：`raw_relaxed_accuracy`、`final_relaxed_accuracy`
- 输出格式质量：`json_valid_rate`
- 可验证证据质量：`verified_rate`、`corrected_rate`、`avg_attempts`
