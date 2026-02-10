# Vector Quantization (VQ) 使用指南

## 概述

Vector Quantization (VQ) 已集成到 GeoT5Gemma 模型中，用于控制位置编码的信息带宽，防止数据泄漏。

## 核心原理

```
位置坐标 → GeoPE → 连续向量 (256维) → VQ → 量化向量 (K个离散选择) → 加到 embedding
                                      ↑
                                  信息瓶颈: log₂(K) bits
```

**关键特性**：
- **信息控制**：K 个 codebook entries → log₂(K) bits 信息量
- **可学习**：codebook 通过 EMA 自适应数据分布
- **防止泄漏**：多个不同坐标映射到同一 entry → 模型无法反推精确坐标

## 配置方式

### 在 config.json 中启用 VQ

```json
{
    "geometric_config": {
        "use_advanced_geo_pe": true,
        "use_vq": true,
        "vq_codebook_size": 256,
        "vq_commitment_cost": 0.25,
        "vq_ema_decay": 0.99,
        "vq_dead_code_threshold": 2,

        "use_geo_self_attn": false,
        "use_geo_cross_attn": false,
        "coord_scale": 1e-6,
        "num_frequencies": 32
    }
}
```

### 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_vq` | bool | false | 是否启用 VQ |
| `vq_codebook_size` | int | 256 | Codebook 大小 (K)，信息量 = log₂(K) bits |
| `vq_commitment_cost` | float | 0.25 | Commitment loss 权重 (β) |
| `vq_ema_decay` | float | 0.99 | EMA 衰减率，越大越稳定但收敛越慢 |
| `vq_dead_code_threshold` | int | 2 | Dead code 复活阈值 |

## Codebook Size 选择建议

| K | log₂(K) | 信息量 | 使用场景 |
|---|---------|--------|---------|
| 16 | 4 bits | 极低 | 测试 / 强信息瓶颈 |
| 64 | 6 bits | 低 | 只需粗略空间感知 |
| 128 | 7 bits | 中低 | **推荐起点** |
| 256 | 8 bits | 中 | **推荐** - 平衡性能和泛化 |
| 512 | 9 bits | 中高 | 需要更细粒度空间信息 |
| 1024+ | 10+ bits | 高 | 可能仍有轻微泄漏风险 |

**选择原则**：
- 从 K=256 开始，观察 training/eval loss
- 如果 loss 仍然过低（< 0.01），减小 K
- 如果 loss 下降太慢或不收敛，增大 K
- **目标**：loss 在合理范围（0.1-1.0），eval loss 与 train loss 接近

## 实验流程

### 1. 基线实验（无 VQ）

```json
{
    "geometric_config": {
        "use_advanced_geo_pe": true,
        "use_vq": false
    }
}
```

预期结果：loss 极低（~0.0005），明显数据泄漏。

### 2. VQ 实验（推荐配置）

```json
{
    "geometric_config": {
        "use_advanced_geo_pe": true,
        "use_vq": true,
        "vq_codebook_size": 256
    }
}
```

预期结果：loss 明显上升到合理范围，模型学习真正的路由逻辑。

### 3. VQ + LARA 组合（最优方案）

```json
{
    "geometric_config": {
        "use_advanced_geo_pe": true,
        "use_vq": true,
        "vq_codebook_size": 128,

        "use_geo_self_attn": true,
        "use_geo_cross_attn": true,
        "enable_encoder_lara": true
    }
}
```

**组合优势**：
- **VQ**: 粗粒度空间区域（log₂(128) = 7 bits）
- **LARA attention bias**: 精细空间关系（scalar/head/pair）
- **互补性强**：VQ 告诉"在哪个区域"，LARA 告诉"距离远近"

## 训练观察指标

### 1. VQ Loss

VQ loss 会加到 total loss 中：
```python
total_loss = task_loss + vq_loss
```

正常的 VQ loss 应该在 0.01-0.1 范围内。

### 2. Codebook Usage

训练过程中检查 codebook 利用率：
```python
model.encoder_pe_vq.usage_count  # 每个 code 的使用频率
used_codes = (model.encoder_pe_vq.usage_count > threshold).sum()
print(f"Used codes: {used_codes}/{codebook_size}")
```

健康的训练应该使用 80%+ 的 codes。如果使用率过低：
- 可能 codebook_size 过大 → 减小 K
- 可能 dead code 机制失效 → 检查 dead_code_threshold

### 3. Loss 曲线对比

| 配置 | Training Loss | Eval Loss | 泛化性 |
|------|---------------|-----------|--------|
| 无 PE | ~0.9 | ~0.9 | 好（但性能受限于 token 解析） |
| GeoPE (无 VQ) | ~0.0005 | ~0.0005 | **差** - 数据泄漏 |
| GeoPE + VQ (K=256) | ~0.3-0.8 | ~0.3-0.8 | 好 - 学到真正逻辑 |
| GeoPE + VQ + LARA | ~0.2-0.6 | ~0.2-0.6 | **最优** |

## 常见问题

### Q1: VQ 后 loss 反而上升了？

**正常现象！** 这说明 VQ 成功创建了信息瓶颈，模型不能再靠记忆精确坐标来"作弊"。Loss 上升到合理范围（如 0.3-0.8）是**好事**，说明模型在学习真正的路由逻辑。

### Q2: 如何确认 VQ 是否工作？

1. 检查模型有 VQ 模块：
```python
assert model.encoder_pe_vq is not None
```

2. 观察 loss：有 VQ 后 loss 应该明显高于无 VQ 的 ~0.0005

3. 检查 codebook 更新：
```python
# 训练前后 codebook 应该变化
initial_codebook = model.encoder_pe_vq.embedding.weight.data.clone()
# ... training ...
final_codebook = model.encoder_pe_vq.embedding.weight.data
diff = (final_codebook - initial_codebook).abs().mean()
print(f"Codebook change: {diff}")  # 应该 > 0
```

### Q3: 如何调参？

**Codebook size (K)**：
- 开始：K=256
- Loss 太低 (< 0.01) → 减小 K (128, 64)
- Loss 太高或不收敛 → 增大 K (512, 1024)

**Commitment cost (β)**：
- 默认 0.25 通常足够
- 如果 GeoPE 不更新（梯度消失）→ 增大到 0.5-1.0
- 如果 codebook 不稳定（剧烈波动）→ 减小到 0.1

**EMA decay**：
- 默认 0.99 足够
- 更稳定但慢收敛 → 0.995-0.999
- 更快适应但不稳定 → 0.95-0.98

### Q4: VQ 和 LARA-only 哪个更好？

**建议组合使用！**

| 方案 | 优点 | 缺点 |
|------|------|------|
| VQ only | 简单、可控、直观 | 只有粗粒度区域信息 |
| LARA only | 精细空间关系 | 可能信息仍然过多 |
| **VQ + LARA** | 互补、最优 | 稍复杂 |

## 代码示例

### 创建启用 VQ 的模型

```python
from transformers import T5GemmaConfig
from flow.models.geo_t5gemma import GeoT5GemmaForConditionalGeneration, GeoConfig

# 配置
config = T5GemmaConfig(...)
geo_config = GeoConfig(
    use_advanced_geo_pe=True,
    use_vq=True,
    vq_codebook_size=256,
    vq_commitment_cost=0.25,
    vq_ema_decay=0.99,
)

# 创建模型
model = GeoT5GemmaForConditionalGeneration(config, geo_config)

# 检查 VQ 模块
print(f"VQ module: {model.encoder_pe_vq}")
print(f"Codebook size: {model.encoder_pe_vq.codebook_size}")
```

### 训练循环

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    # Forward (VQ loss 会自动加到 total loss)
    outputs = model(
        input_ids=batch['input_ids'],
        labels=batch['labels'],
        encoder_abs_positions=batch['encoder_abs_positions'],
        encoder_rel_positions=batch['encoder_rel_positions'],
    )

    loss = outputs.loss  # 包含 task_loss + vq_loss

    # Backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # 监控 codebook 使用
    if step % 100 == 0:
        used = (model.encoder_pe_vq.usage_count > 0.1).sum()
        print(f"Loss: {loss.item():.4f}, Used codes: {used}/{K}")
```

## 验证泛化性

```python
# 在完全不同的 design 上测试
eval_loss_same_design = evaluate(model, same_design_eval_loader)
eval_loss_new_design = evaluate(model, new_design_eval_loader)

print(f"Same design eval loss: {eval_loss_same_design:.4f}")
print(f"New design eval loss: {eval_loss_new_design:.4f}")

# 健康的泛化：两者接近
# 数据泄漏：same design 极低，new design 很高
```

## 总结

✅ **VQ 是解决位置编码数据泄漏的有效方法**
✅ **推荐配置：K=256, β=0.25, decay=0.99**
✅ **最佳实践：VQ + LARA 组合使用**
✅ **关键指标：loss 从 ~0.0005 上升到 ~0.3-0.8**

---

如有问题，请参考 `test_vq_integration.py` 中的测试用例。
