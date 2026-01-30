# LieAlgebraRelativeAttention (LARA) 集成到 T5Gemma 的完整方案

## 当前状态分析

### 已实现的部分
1. ✅ **LieAlgebraRelativeAttention** (flow/models/geometric_attention.py:285-471)
   - 完整的几何注意力实现
   - 基于四元数的 GeometricPositionEmbedding
   - 几何偏置 MLP

2. ✅ **位置编码集成** (flow/models/geo_t5gemma.py)
   - FourierPositionEmbedding
   - GeometryAwarePositionEmbedding
   - 在 input embeddings 层面添加位置信息

3. ✅ **配置支持** (config.json:58)
   - `enable_geometric_attention: true`
   - 完整的几何配置参数

### 缺失的部分
❌ **Attention Layer 替换**
   - 当前：coordinates 只添加到 input embeddings
   - 需要：coordinates 传递到每个 attention layer
   - 目标：用 LARA 替换 T5GemmaSelfAttention

---

## T5Gemma 架构分析

### 关键组件接口

```python
# 1. 模型顶层
T5GemmaForConditionalGeneration.forward(
    input_ids, attention_mask,
    inputs_embeds,              # 你已经在这里添加了 position embeddings
    decoder_input_ids, ...
) → Seq2SeqLMOutput

# 2. Encoder
T5GemmaEncoder.forward(
    input_ids, attention_mask,
    position_ids,              # 用于计算 position_embeddings
    inputs_embeds, ...
) → BaseModelOutput

# 3. EncoderLayer (重复 N 层)
T5GemmaEncoderLayer.forward(
    hidden_states,
    position_embeddings,       # tuple(cos, sin) for RoPE
    attention_mask,
    position_ids, ...
) → (hidden_states, attention_weights)

# 4. SelfAttention
T5GemmaSelfAttention.forward(
    hidden_states,
    position_embeddings,       # tuple(cos, sin) for RoPE
    attention_mask,
    past_key_value,
    cache_position, ...
) → (attn_output, attn_weights, present_key_value)
```

### 数据流对比

**当前实现（仅 Position Embedding）:**
```
input_ids → embeddings → [+ FourierPE] → encoder → decoder → output
                            ↑
                      coordinates
```

**目标实现（LARA Attention）:**
```
input_ids → embeddings → [+ FourierPE] → encoder → decoder → output
                            ↑                ↑
                      coordinates      coordinates (每层)
                                           ↓
                                    LARA Attention
```

---

## 集成方案设计

### 方案 1: 最小侵入式（推荐）

**核心思路**: 创建自定义的 Layer 类，在保持接口兼容的同时替换 attention

#### Step 1: 创建 LARA Wrapper

```python
# flow/models/geometric_attention.py

class T5GemmaLARAAttention(nn.Module):
    """
    LARA Attention wrapper compatible with T5GemmaSelfAttention interface.

    Key differences from standard LARA:
    - Accepts position_embeddings (ignored) for interface compatibility
    - Accepts coordinates via kwargs
    - Supports past_key_value for caching
    - Returns format: (attn_output, attn_weights, present_key_value)
    """

    def __init__(
        self,
        config,
        layer_idx: int = 0,
        coord_scale: float = 1e-5,
        use_geometric_bias: bool = True,
        bias_mlp_hidden: int = 64,
    ):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.layer_idx = layer_idx

        # Core LARA components
        self.lara = LieAlgebraRelativeAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=config.dropout_rate,
            use_geometric_bias=use_geometric_bias,
            bias_mlp_hidden=bias_mlp_hidden,
            coord_scale=coord_scale,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],  # Ignored, for compatibility
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        coordinates: Optional[torch.Tensor] = None,  # NEW: (batch, seq_len, 3)
        output_attentions: bool = False,
        **kwargs
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """
        Forward pass compatible with T5GemmaSelfAttention.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            position_embeddings: Ignored (RoPE not used in LARA)
            attention_mask: (batch, seq_len) or (batch, seq_len, seq_len)
            past_key_value: Cache (not yet supported in LARA)
            cache_position: Cache position indices
            coordinates: (batch, seq_len, 3) - Required for LARA
            output_attentions: Whether to return attention weights

        Returns:
            (attn_output, attn_weights, present_key_value)
        """

        if coordinates is None:
            raise ValueError(
                "LieAlgebraRelativeAttention requires 'coordinates' parameter. "
                "Make sure to pass encoder_abs_positions through the model."
            )

        # TODO: Implement KV caching support if needed
        if past_key_value is not None:
            raise NotImplementedError(
                "KV caching not yet supported in LieAlgebraRelativeAttention"
            )

        # Call LARA
        lara_output = self.lara(
            hidden_states,
            coordinates,
            attention_mask,
            output_attentions=output_attentions
        )

        if output_attentions:
            attn_output, attn_weights = lara_output
        else:
            attn_output = lara_output[0]
            attn_weights = None

        # Return format compatible with T5GemmaSelfAttention
        present_key_value = None  # No caching yet
        return attn_output, attn_weights, present_key_value
```

#### Step 2: 创建自定义 EncoderLayer

```python
# flow/models/geo_t5gemma.py

from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoderLayer
from .geometric_attention import T5GemmaLARAAttention

class GeoT5GemmaEncoderLayer(T5GemmaEncoderLayer):
    """
    T5GemmaEncoderLayer with optional LARA geometric attention.

    Extends the base layer to:
    1. Optionally replace self_attn with LARA
    2. Accept and pass coordinates to attention layer
    """

    def __init__(self, config, layer_idx: int, enable_geometric_attention: bool = False):
        super().__init__(config, layer_idx)

        # Replace self-attention with LARA if enabled
        if enable_geometric_attention:
            geo_config = getattr(config, 'geometric_config', {})
            self.self_attn = T5GemmaLARAAttention(
                config,
                layer_idx=layer_idx,
                coord_scale=geo_config.get('coord_scale', 1e-5),
                use_geometric_bias=geo_config.get('use_geometric_bias', True),
                bias_mlp_hidden=geo_config.get('bias_mlp_hidden', 64),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        coordinates: Optional[torch.Tensor] = None,  # NEW parameter
        **kwargs
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward with coordinate support for LARA.

        Args:
            coordinates: (batch, seq_len, 3) - 3D coordinates for geometric attention
        """

        # Self-Attention Block
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)

        # Pass coordinates to attention layer
        hidden_states, self_attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            coordinates=coordinates,  # Pass coordinates
            output_attentions=output_attentions,
            **kwargs
        )

        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        # MLP Block (unchanged)
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
```

#### Step 3: 创建自定义 Encoder

```python
# flow/models/geo_t5gemma.py

from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoder

class GeoT5GemmaEncoder(T5GemmaEncoder):
    """
    T5GemmaEncoder with LARA support.

    Extends the base encoder to:
    1. Use GeoT5GemmaEncoderLayer instead of T5GemmaEncoderLayer
    2. Accept and pass coordinates through layer stack
    """

    def __init__(self, config):
        # Don't call super().__init__ to avoid creating base layers
        nn.Module.__init__(self)

        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        # Create custom layers with LARA support
        geo_config_dict = getattr(config, 'geometric_config', {})
        enable_geo_attn = geo_config_dict.get('enable_geometric_attention', False)

        self.layers = nn.ModuleList([
            GeoT5GemmaEncoderLayer(
                config,
                layer_idx,
                enable_geometric_attention=enable_geo_attn
            )
            for layer_idx in range(config.encoder.num_hidden_layers)
        ])

        self.final_layer_norm = T5GemmaRMSNorm(config.hidden_size)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        coordinates: Optional[torch.Tensor] = None,  # NEW parameter
        **flash_attn_kwargs
    ) -> BaseModelOutput:
        """
        Forward with coordinate support.

        Args:
            coordinates: (batch, seq_len, 3) - Passed to each layer
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = self.dropout(inputs_embeds)

        # Prepare position embeddings (for non-LARA layers or compatibility)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Layer stack
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    position_ids,
                    output_attentions,
                    coordinates,  # Pass coordinates
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    coordinates=coordinates,  # Pass coordinates
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[1],)

        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
```

#### Step 4: 修改 GeoT5GemmaForConditionalGeneration

```python
# flow/models/geo_t5gemma.py

class GeoT5GemmaForConditionalGeneration(T5GemmaForConditionalGeneration):
    # ... (现有代码保持不变)

    def __init__(
        self,
        config: T5GemmaConfig,
        geo_config: Optional[Union[GeoConfig, Dict[str, Any]]] = None
    ):
        # Call parent (不初始化 geometric modules，先创建基础结构)
        super(T5GemmaForConditionalGeneration, self).__init__(config)

        # Parse geo_config
        if geo_config is None:
            self.geo_config = GeoConfig()
        elif isinstance(geo_config, dict):
            self.geo_config = GeoConfig.from_dict(geo_config)
        else:
            self.geo_config = geo_config

        # Store geo_config in model config for layer access
        config.geometric_config = self.geo_config.to_dict()

        # Replace encoder with GeoT5GemmaEncoder (if LARA enabled)
        if self.geo_config.enable_geometric_attention:
            self.encoder = GeoT5GemmaEncoder(config)
            # Decoder can also be replaced similarly if needed

        # Initialize Position Embedding modules (existing code)
        self.encoder_geo_pe = None
        self.decoder_geo_pe = None
        # ... (rest of existing __init__ code)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        # ... (existing parameters)
        encoder_abs_positions: Optional[torch.Tensor] = None,
        encoder_rel_positions: Optional[torch.Tensor] = None,
        decoder_coordinates: Optional[torch.Tensor] = None,
        encoder_coordinates: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Handle legacy input
        if encoder_abs_positions is None and encoder_coordinates is not None:
            encoder_abs_positions = encoder_coordinates

        # Get embeddings and add position embeddings (existing code)
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if inputs_embeds is not None:
            inputs_embeds = self._add_encoder_geometric_embeddings(
                inputs_embeds,
                encoder_abs_positions,
                encoder_rel_positions,
                attention_mask,
            )

        # === KEY CHANGE: Pass coordinates to encoder ===
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=None,  # Already embedded
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                coordinates=encoder_abs_positions,  # Pass coordinates to layers
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        # Decoder (existing code, unchanged)
        # ...

        # Rest of forward remains the same
        return super(T5GemmaForConditionalGeneration, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=None,
            decoder_inputs_embeds=decoder_inputs_embeds,
            # ... rest of parameters
        )
```

---

## 实现步骤清单

### Phase 1: 核心集成 (必须)

- [ ] **Step 1.1**: 在 `geometric_attention.py` 中添加 `T5GemmaLARAAttention`
- [ ] **Step 1.2**: 在 `geo_t5gemma.py` 中添加 `GeoT5GemmaEncoderLayer`
- [ ] **Step 1.3**: 在 `geo_t5gemma.py` 中添加 `GeoT5GemmaEncoder`
- [ ] **Step 1.4**: 修改 `GeoT5GemmaForConditionalGeneration.__init__` 以替换 encoder
- [ ] **Step 1.5**: 修改 `GeoT5GemmaForConditionalGeneration.forward` 以传递 coordinates
- [ ] **Step 1.6**: 测试基本 forward pass (without generation)

### Phase 2: Decoder 支持 (可选)

- [ ] **Step 2.1**: 创建 `GeoT5GemmaDecoderLayer` (类似 EncoderLayer)
- [ ] **Step 2.2**: 创建 `GeoT5GemmaDecoder`
- [ ] **Step 2.3**: 处理 decoder coordinates (如果需要)

### Phase 3: 高级功能 (可选)

- [ ] **Step 3.1**: 实现 KV caching 支持 (用于 generation)
- [ ] **Step 3.2**: 实现 Grouped Query Attention (GQA) 支持
- [ ] **Step 3.3**: 添加 Flash Attention 支持
- [ ] **Step 3.4**: Gradient checkpointing 兼容性测试

### Phase 4: 测试与优化

- [ ] **Step 4.1**: 单元测试 (attention layer 输出维度)
- [ ] **Step 4.2**: 集成测试 (完整 forward + backward)
- [ ] **Step 4.3**: Generation 测试 (beam search, sampling)
- [ ] **Step 4.4**: 性能 benchmark (vs standard T5Gemma)

---

## 关键注意事项

### 1. Coordinate 维度检查
```python
# 在 forward 中添加断言
assert coordinates is not None, "LARA requires coordinates"
assert coordinates.shape == (batch_size, seq_len, 3), \
    f"Expected coordinates shape (B, L, 3), got {coordinates.shape}"
```

### 2. Attention Mask 处理
```python
# LARA 目前期望 (batch, seq_len) 或 (batch, seq_len, seq_len)
# T5Gemma 可能使用不同格式，需要转换
if attention_mask.dim() == 4:  # (batch, 1, seq_len, seq_len)
    attention_mask = attention_mask.squeeze(1)
```

### 3. Head Dimension 约束
```python
# GeometricPositionEmbedding 假设 head_dim % 3 == 0 (或有 padding)
# config.json 中 head_dim = 64
# 64 % 3 = 1 (需要 padding 到 63)
# 已在 GeometricPositionEmbedding.__init__ 中处理
```

### 4. Gradient Checkpointing
```python
# 确保 coordinates 参数正确传递
# 在 GeoT5GemmaEncoder.forward 中:
layer_outputs = self._gradient_checkpointing_func(
    encoder_layer.__call__,
    hidden_states,
    position_embeddings,
    attention_mask,
    position_ids,
    output_attentions,
    coordinates,  # 必须显式传递
)
```

### 5. Mixed Precision Training
```python
# LARA 中的 quaternion 计算对数值稳定性敏感
# 在 geometric_attention.py 中确保 float32 计算:
coords = coordinates.float() * self.coord_scale
```

---

## 测试代码示例

```python
# test_lara_integration.py

import torch
from flow.models.geo_t5gemma import (
    GeoT5GemmaForConditionalGeneration,
    create_geo_t5gemma_config,
    GeoConfig,
)

def test_lara_forward():
    """测试 LARA 集成的 forward pass"""

    # Create config
    config = create_geo_t5gemma_config(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=64,
    )

    geo_config = GeoConfig(
        enable_fourier_pe=False,
        enable_geometry_aware_pe=True,
        enable_geometric_attention=True,  # Enable LARA
        coord_scale=1e-5,
    )

    # Create model
    model = GeoT5GemmaForConditionalGeneration(config, geo_config)
    model.eval()

    # Prepare inputs
    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # Coordinates (absolute positions)
    encoder_abs_positions = torch.randn(batch_size, seq_len, 3) * 10000
    encoder_rel_positions = torch.randn(batch_size, seq_len, 3) * 1000

    labels = torch.randint(0, 1000, (batch_size, seq_len))

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_abs_positions=encoder_abs_positions,
            encoder_rel_positions=encoder_rel_positions,
            labels=labels,
        )

    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Logits shape: {outputs.logits.shape}")
    assert outputs.logits.shape == (batch_size, seq_len, 1000)
    print("✓ LARA forward pass successful!")

if __name__ == "__main__":
    test_lara_forward()
```

---

## 性能优化建议

### 1. 缓存 Quaternion 计算
```python
# 在 GeometricPositionEmbedding 中
# 如果 coordinates 不变，可以缓存 (w, qx, qy, qz)
self._coord_cache = None
self._quaternion_cache = None
```

### 2. 稀疏 Geometric Bias
```python
# 对于长序列，geometric_bias 是 O(L²) 内存
# 考虑稀疏化或分块计算
if seq_len > 512:
    # 使用 sparse attention 或 local attention
    pass
```

### 3. Mixed Precision
```python
# 使用 autocast 但保持关键计算在 float32
with torch.cuda.amp.autocast():
    # 大部分计算
    pass

# Quaternion 计算强制 float32
coords = coordinates.float()
```

---

## FAQ

### Q1: 为什么不直接修改 transformers 源码？
**A**: 保持与上游兼容性，便于升级。使用继承和组合模式更灵活。

### Q2: Decoder 是否需要 LARA？
**A**: 可选。Decoder 是自回归的，主要依赖 cross-attention。如果需要，可以类似地替换。

### Q3: 如何处理 generation 时的 coordinates？
**A**: 在 `prepare_inputs_for_generation` 中确保 coordinates 被传递，且不随生成步数改变（因为是encoder端的坐标）。

### Q4: LARA 是否支持 FlashAttention？
**A**: 目前未实现。可以参考 `xformers` 或自定义 CUDA kernel 加速 geometric bias 计算。

### Q5: 如何验证 LARA 确实被使用？
**A**:
```python
# 检查模型结构
for name, module in model.named_modules():
    if 'self_attn' in name:
        print(f"{name}: {type(module)}")
# 应该看到 T5GemmaLARAAttention
```

---

## 相关文件

- **实现**: `flow/models/geometric_attention.py`, `flow/models/geo_t5gemma.py`
- **配置**: `config.json`, `flow/config.py`
- **测试**: 待创建 `tests/test_lara_integration.py`
- **文档**: 本文件

---

## 总结

集成 LARA 到 T5Gemma 的核心步骤：

1. **创建适配器层** (`T5GemmaLARAAttention`) - 包装 LARA 使其兼容 T5Gemma 接口
2. **自定义 EncoderLayer** - 替换 self_attn 并传递 coordinates
3. **自定义 Encoder** - 使用自定义 layer 并传递 coordinates
4. **修改模型 forward** - 确保 coordinates 从顶层传递到各个 layer

**关键挑战**:
- 接口兼容性 (position_embeddings vs coordinates)
- KV caching 支持
- Gradient checkpointing 兼容

**下一步**: 按照 Phase 1 实现核心集成代码。
