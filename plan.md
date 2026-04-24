# Migration Plan: T5-Gemma → Llama 3 (Decoder-Only)

## Background

Current architecture: T5-Gemma (encoder-decoder)
- source tokens → encoder
- decoder auto-regressively generates target tokens from `decoder_start_token_id`

Target architecture: Llama 3 (decoder-only / causal LM)
- source + target tokens concatenated into a single sequence
- loss only computed on target tokens (source positions masked to -100)

## Core Paradigm Shift

```
T5-Gemma:  [source] → encoder → cross-attention → decoder → [target]
Llama 3:   [source | target] → causal LM → next-token prediction (loss on target only)
```

Generation at inference:
```
T5-Gemma:  generate(input_ids=source, ...)  →  outputs = target tokens directly
Llama 3:   generate(input_ids=source, ...)  →  outputs = source + target, must slice [:, src_len:]
```

---

## Files to Change (in order)

### Step 1: `flow/config.py` — TrainingModel
**Status: [x] done**

| Change | Detail |
|--------|--------|
| Remove `sliding_window` | Llama 3 uses full attention, not sliding window |
| Remove `dropout_rate` | Llama uses `attention_dropout` instead |
| Add `rope_theta` | Llama 3 default: 500000.0 |
| Add `rms_norm_eps` | Llama 3 default: 1e-5 |

### Step 2: `flow/training/pipeline.py` — TrainingPipeline
**Status: [x] done**

| Change | Detail |
|--------|--------|
| Replace imports | `T5GemmaConfig/ForConditionalGeneration/ModuleConfig` → `LlamaConfig`, `LlamaForCausalLM` |
| Replace `Seq2SeqTrainer/Arguments` | → `Trainer`, `TrainingArguments` |
| Replace `DataCollatorForSeq2Seq` | → `DataCollatorForLanguageModeling(mlm=False)` |
| Rewrite `_initialize_T5Gemma_model` | Single `LlamaConfig` (no encoder/decoder split) → `LlamaForCausalLM` |
| Rewrite `tokenize_sample` | Concatenate source+target; set source positions in labels to -100 |
| Update `_setup_training_arguments` | `Seq2SeqTrainingArguments` → `TrainingArguments` |
| Update `_initialize_trainer` | `Seq2SeqTrainer` → `Trainer` |

Critical change in `tokenize_sample`:
```python
# Before (encoder-decoder)
input_ids = source_encs["input_ids"]
labels = [-100 if t == pad_id else t for t in target_ids]

# After (decoder-only)
input_ids = source_ids + target_ids            # concatenated
labels = [-100] * len(source_ids) + target_ids # mask source, keep target
# total length capped at max_src_len + max_tgt_len
```

### Step 3: `flow/launch_evaluation.py` — Inference
**Status: [x] done**

| Change | Detail |
|--------|--------|
| Replace model import/type hint | `T5GemmaForConditionalGeneration` → `LlamaForCausalLM` |
| Update `load_components` | Load `LlamaForCausalLM` instead of `T5GemmaForConditionalGeneration` |
| Rewrite `collect_fn` | Left-pad source tokens (required for decoder-only batch inference) |
| Remove `decoder_start_token_id` from GenerationConfig | Not used in causal LM |
| Slice output after generation | `outputs[:, input_len:]` to strip the input prefix |

Critical change in inference:
```python
# Before: output is target tokens directly
preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# After: output includes input prefix, must slice
input_len = batch["input_ids"].shape[1]
preds = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
```

Left-padding in `collect_fn`:
```python
# Before: default right-padding (fine for encoder input)
tokenizer(..., padding="max_length")

# After: left-padding (required so all generated tokens are at the right end)
tokenizer.padding_side = "left"
tokenizer(..., padding=True)
```

---

## Change Log

| Date | File | Change Summary |
|------|------|----------------|
| 2026-04-24 | `flow/config.py` | `TrainingModel`: 删除 `sliding_window`/`dropout_rate`，新增 `rope_theta`(500000.0)、`rms_norm_eps`(1e-5)、`attention_dropout`(0.0) |
| 2026-04-24 | `flow/training/pipeline.py` | imports: T5GemmaConfig/ForConditionalGeneration/ModuleConfig → LlamaConfig/LlamaForCausalLM；Seq2SeqTrainer/Arguments → Trainer/TrainingArguments；DataCollatorForSeq2Seq → DataCollatorForLanguageModeling |
| 2026-04-24 | `flow/training/pipeline.py` | `tokenize_sample` 重写：source+target 拼接为单一序列，labels 中 source 部分置为 -100 |
| 2026-04-24 | `flow/training/pipeline.py` | `_initialize_T5Gemma_model` → `_initialize_llama_model`：单一 LlamaConfig 替换 encoder/decoder 双配置，LlamaForCausalLM 替换 T5GemmaForConditionalGeneration |
| 2026-04-24 | `flow/launch_evaluation.py` | model import/load: T5GemmaForConditionalGeneration → LlamaForCausalLM |
| 2026-04-24 | `flow/launch_evaluation.py` | `collect_fn`: 改为 left-padding（decoder-only batch inference 必须） |
| 2026-04-24 | `flow/launch_evaluation.py` | `GenerationConfig`: 删除 `decoder_start_token_id` |
| 2026-04-24 | `flow/launch_evaluation.py` | 推理输出截断：`outputs[:, input_len:]` 去掉 input prefix |
| 2026-04-24 | `flow/training/pipeline.py` | **Review fix #1**: `DataCollatorForLanguageModeling(mlm=False)` 会用 `input_ids.clone()` 覆盖 labels，摧毁 prefix-LM 的 source 掩码 → 改用 `DataCollatorForSeq2Seq(label_pad_token_id=-100)`，正确保留 tokenize_sample 设置的 labels |
| 2026-04-24 | `flow/config.py` | **Review fix #2**: `max_position_embeddings` 默认 512 < `max_src_len + max_tgt_len = 1024`，decoder-only 拼接后会越界 → 默认值改为 1024，并添加注释说明约束 |
| 2026-04-24 | `flow/training/pipeline.py` | **Review fix #2**: `_initialize_llama_model` 添加 assert：`max_position_embeddings >= max_src_len + max_tgt_len`，配置错误时提前报错而不是运行时 IndexError |
| 2026-04-24 | `flow/launch_evaluation.py` | **Review fix #3**: `tokenizer.padding_side = "left"` 从 `collect_fn` 移到 `load_components`，避免每 batch 重复设置全局状态 |
| 2026-04-24 | `flow_config_llama.json` | 新建 Llama 版 config，模型 size 对齐 `flow_config.json`（hidden=512, layers=6, heads=8, kv_heads=4, head_dim=64）；`max_position_embeddings` 从 256 → 512（必须 ≥ `max_src_len + max_tgt_len = 384`）；`max_new_tokens` 从 512 → 256（对齐 `max_tgt_len`，训练分布一致）；路径前缀 `Medium-Refine` → `Medium-Refine-Llama` 避免覆盖 T5-Gemma 产物 |
| 2026-04-24 | `flow_config_llama.json` + `baseline/flow_config.json` | **Runtime fix**: 训练报错 `Expected input batch_size (18944) to match target batch_size (37888)` = 148×128 vs 148×256，说明加载了旧 T5-Gemma 格式的 split_dataset（input_ids=source only, labels=target only）。根因：`split_dataset_dir` 指向 liweiguo 的 T5-Gemma 缓存，被 `_load_or_create_dataset` 的存在性检查误用 → 改为 llama 专属路径强制重建；同时 `num_hidden_layers: 6 → 12` 让参数量 23.71M → 47.31M，接近 T5-Gemma 的 52.23M（差距来自 Llama 无 cross-attention，属正常） |
| 2026-04-24 | `flow/training/pipeline.py` | **Runtime fix v2**: 修改 split_dataset_dir 后仍然 shape mismatch。根因：`dataset.map()` 命中了 liweiguo `token_dataset/` 目录下 T5-Gemma 时代遗留的 `cache-*.arrow` 文件（HF datasets 基于 fingerprint 缓存 map 结果），新版 `tokenize_sample` 被旧缓存直接替换输出 → 给 `dataset.map(...)` 加 `load_from_cache_file=False`，强制每次重算。同时把 `Medium-Refine-Llama/stage_training/split_dataset` 重命名为 `.bad_cache_20260424` 备份，触发下一轮重建 |

---

## Notes

- `launch_tokenization.py`: **no changes needed** — tokenizer training is model-agnostic; `source_tokens` and `target_tokens` remain as separate columns for use in training pipeline
- `max_src_len` + `max_tgt_len` in config still valid; training pipeline uses their sum as total sequence budget
- Beam search parameters (`num_beams`, `max_new_tokens`, etc.) in `EvaluationGeneration` config are unchanged
- Adafactor optimizer: still compatible with `LlamaForCausalLM`; AdamW and Lion also unchanged
- `tie_word_embeddings=True` 沿用自 T5-Gemma（Llama 3 原版默认 False）：从头训练小模型时节省参数，是刻意保留的设计决策

## Post-Migration Sanity Checks

在正式跑完整 pipeline 前，建议先做以下验证：

- [ ] 跑 1 个 training batch，打印 `(labels != -100).sum() / labels.numel()`，确认 source 位置已被掩码（期望值约等于 `max_tgt_len / (max_src_len + max_tgt_len)`）
- [ ] 打印模型参数量，对比 T5-Gemma 版本：同 `hidden_size/num_hidden_layers` 下 decoder-only 约为 encoder-decoder 的 ~50%
- [ ] 评估时打印单样本 `input_ids.shape[1]` 和 `outputs.shape[1]`，确认 `outputs - input_ids` 长度 ≤ `max_new_tokens`
- [ ] 确认 `LlamaConfig` 在当前 transformers 版本下支持 `head_dim` 参数（Llama 3.1+ 才显式支持；较老版本会从 `hidden_size/num_attention_heads` 推导）
