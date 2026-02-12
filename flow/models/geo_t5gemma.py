#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   geo_t5gemma.py
@Time    :   2025/01/14
@Author  :   Dawn Li
@Version :   1.0
@Desc    :   GeoT5Gemma: T5Gemma with Geometry-Aware Position Embeddings.
             Combined implementation of Steps 2 and 3 for geometry-aware
             trajectory generation in EDA routing.

             This model extends T5GemmaForConditionalGeneration with:
             - Fourier Position Embedding (Step 2): Added to input embeddings
             - Geometric Attention (Step 3): Optional LARA-style attention

             Reference:
             - Base: T5GemmaForConditionalGeneration from HuggingFace
             - FoPE: "Fourier Position Embedding" (ICML 2024)
             - GeoPE: "GeoPE: A Unified Geometric Positional Embedding" (Arxiv 2025)
"""

from dataclasses import dataclass, field
from torch.nn.modules.module import Module
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    T5GemmaConfig,
    T5GemmaForConditionalGeneration,
    T5GemmaModuleConfig,
)
from transformers.models.t5gemma.modeling_t5gemma import (
    T5GemmaEncoderLayer,
    T5GemmaEncoder,
    T5GemmaDecoderLayer,
    T5GemmaDecoder,
    T5GemmaRMSNorm,
)
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions

from .position_embedding import (
    FourierPositionEmbedding,
    GeometryAwarePositionEmbedding,
    GeometryAwarePositionEmbeddingTMP,
    GeoPEConfig,
)
from .geometric_attention import (
    LieAlgebraRelativeAttention,
    T5GemmaLARAAttention,
    CrossLARAAttention,
)


@dataclass
class GeoConfig:
    """
    Configuration for geometry-aware embeddings.

    This configuration controls all geometric embedding components:
    - Coordinate handling and scaling
    - Fourier Position Embedding parameters (simple 3D Fourier)
    - Geometry-Aware Position Embedding parameters (advanced: XY Fourier + Metal Layer + Polar Rel)
    - Geometric Attention (LARA) parameters

    Position Embedding Design:
    - Only semantic tokens (<DRIVER>, <LOAD>) receive position embeddings
    - <DRIVER>: Absolute position encoding via Fourier
    - <LOAD>: Absolute position + Relative position (from driver)
    - Other tokens: Zero position embedding

    Attributes:
        use_basic_fourier_pe: Whether to use simple Fourier Position Embedding
        use_advanced_geo_pe: Whether to use advanced Geometry-Aware PE (overrides use_basic_fourier_pe)
        use_geo_self_attn: Whether to use LARA geometric attention
        coord_scale: Scaling factor for coordinates (chip coords are large)
        num_frequencies: Number of frequency bands for Fourier embedding
        num_harmonics: Number of circular harmonics for direction encoding
        max_metal_layers: Maximum number of metal layers
        max_layer_delta: Maximum layer difference for relative encoding
        max_wavelength: Maximum wavelength for frequency bands
        min_wavelength: Minimum wavelength for frequency bands
        learnable_fourier_coefficients: Whether Fourier coefficients are learnable
        separate_sin_cos_basis: Whether to use separate sin/cos coefficient matrices
        floor_freq_ratio: Ratio for clipping low frequencies
        max_sequence_length: Maximum sequence length for frequency clipping
        pe_dropout: Dropout rate for position embeddings
        use_geometric_bias: Whether to use MLP bias in LARA
        bias_mlp_hidden: Hidden dimension for geometric bias MLP
    """

    # General settings
    use_basic_fourier_pe: bool = False  # Simple 3D Fourier (deprecated)
    use_advanced_geo_pe: bool = True  # Advanced Geometry-Aware PE (recommended)
    use_metal_layer_only_pe: bool = False  # Only Metal Layer PE (for testing metal layer effects)
    use_geo_self_attn: bool = False  # Enable LARA for decoder self-attention
    use_geo_cross_attn: bool = False  # Enable LARA for cross-attention
    enable_encoder_lara: bool = False  # Enable LARA for encoder (usually not recommended)
    coord_scale: float = 1e-5  # Smaller scale for large chip coordinates

    # Fourier / Geometry-Aware Position Embedding settings
    num_frequencies: int = 32  # Frequency bands for Fourier encoding
    num_harmonics: int = 8  # Circular harmonics for direction encoding
    max_metal_layers: int = 16  # Maximum metal layers (typically 10-15)
    max_layer_delta: int = 10  # Maximum layer difference
    max_wavelength: float = 10000.0
    min_wavelength: float = 1.0
    learnable_fourier_coefficients: bool = True
    separate_sin_cos_basis: bool = True
    floor_freq_ratio: float = 1.0
    max_sequence_length: int = 512
    pe_dropout: float = 0.1  # Dropout for position embeddings

    # Geometric Attention settings (Step 3)
    use_geometric_bias: bool = True
    bias_mlp_hidden: int = 64

    # Vector Quantization settings (Information Bottleneck)
    use_vq: bool = False  # Whether to apply VQ to position embeddings
    vq_codebook_size: int = 256  # Number of codebook entries (K), info: log2(K) bits
    vq_commitment_cost: float = 0.25  # β for commitment loss
    vq_ema_decay: float = 0.99  # EMA decay for codebook updates
    vq_dead_code_threshold: int = 2  # Usage threshold for dead code revival

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GeoConfig":
        """Create GeoConfig from dictionary."""
        return cls(**{
            k: v for k, v in config_dict.items()
            if k in cls.__dataclass_fields__
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }

    def to_geope_config(self, hidden_size: int) -> GeoPEConfig:
        """Convert to GeoPEConfig for GeometryAwarePositionEmbedding."""
        return GeoPEConfig(
            hidden_size=hidden_size,
            num_frequencies=self.num_frequencies,
            num_harmonics=self.num_harmonics,
            max_metal_layers=self.max_metal_layers,
            max_layer_delta=self.max_layer_delta,
            coord_scale=self.coord_scale,
            max_wavelength=self.max_wavelength,
            min_wavelength=self.min_wavelength,
            learnable_frequencies=self.learnable_fourier_coefficients,
            dropout=self.pe_dropout,
        )


class GeoT5GemmaEncoderLayer(T5GemmaEncoderLayer):
    """
    T5GemmaEncoderLayer with optional LARA geometric attention.

    Extends the base T5GemmaEncoderLayer to:
    1. Optionally replace self_attn with T5GemmaLARAAttention
    2. Accept and pass coordinates parameter to attention layer

    The layer structure remains the same:
    - Pre-LayerNorm → Self-Attention → Post-LayerNorm → Residual
    - Pre-LayerNorm → MLP → Post-LayerNorm → Residual

    Args:
        config: T5GemmaConfig or T5GemmaModuleConfig
        layer_idx: Layer index in the encoder
        use_geo_self_attn: Whether to use LARA instead of standard attention
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        use_geo_self_attn: bool = False,
        geo_config_dict: Optional[Dict[str, Any]] = None,
    ):
        # Initialize base layer (creates standard self_attn)
        super().__init__(config, layer_idx)

        # Replace self-attention with LARA if enabled
        if use_geo_self_attn:
            # Get geometric config parameters
            if geo_config_dict is None:
                geo_config_dict = {}

            self.self_attn = T5GemmaLARAAttention(
                config,
                layer_idx=layer_idx,
                coord_scale=geo_config_dict.get('coord_scale', 1e-5),
                use_geometric_bias=geo_config_dict.get('use_geometric_bias', True),
                bias_mlp_hidden=geo_config_dict.get('bias_mlp_hidden', 64),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        coordinates: Optional[torch.Tensor] = None,  # NEW parameter
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass with coordinate support for LARA.

        Args:
            hidden_states: Input hidden states (batch, seq_len, hidden_size)
            position_embeddings: RoPE embeddings (cos, sin tuples)
            attention_mask: Attention mask
            position_ids: Position indices
            output_attentions: Whether to return attention weights
            coordinates: 3D coordinates (batch, seq_len, 3) - Used by LARA

        Returns:
            Tuple of (hidden_states, attention_weights)
        """

        # Self-Attention Block
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)

        # Pass coordinates to attention layer (used by LARA, ignored by standard attention)
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

        # MLP Block (unchanged from base class)
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class GeoT5GemmaEncoder(T5GemmaEncoder):
    """
    T5GemmaEncoder with LARA geometric attention support.

    Extends the base T5GemmaEncoder to:
    1. Use GeoT5GemmaEncoderLayer instead of T5GemmaEncoderLayer
    2. Accept and pass coordinates parameter through the layer stack

    Args:
        config: T5GemmaConfig with geometric_config attribute
    """

    def __init__(self, config):
        # Call nn.Module.__init__ directly (skip T5GemmaEncoder.__init__)
        nn.Module.__init__(self)

        # Get encoder config (handle both T5GemmaConfig and T5GemmaModuleConfig)
        if hasattr(config, 'encoder'):
            encoder_config = config.encoder
        else:
            encoder_config = config

        # Store encoder config (not full config) - required by parent methods
        self.config = encoder_config
        # Also store full config for access to geometric_config, vocab_size, etc.
        self._full_config = config

        # Embedding layer
        self.embed_tokens = nn.Embedding(
            self._full_config.vocab_size,
            encoder_config.hidden_size,
            padding_idx=self._full_config.pad_token_id,
        )

        # Create custom encoder layers with optional LARA support
        geo_config_dict = getattr(self._full_config, 'geometric_config', {})
        enable_geo_attn = geo_config_dict.get('use_geo_self_attn', False)

        # Pass encoder_config to layers (not the full config)
        self.layers = nn.ModuleList([
            GeoT5GemmaEncoderLayer(
                encoder_config,  # Pass encoder_config, not full config
                layer_idx=layer_idx,
                use_geo_self_attn=enable_geo_attn,
                geo_config_dict=geo_config_dict,  # Pass geo_config_dict
            )
            for layer_idx in range(encoder_config.num_hidden_layers)
        ])

        self.final_layer_norm = T5GemmaRMSNorm(
            encoder_config.hidden_size,
            eps=getattr(encoder_config, 'rms_norm_eps', 1e-6)
        )

        self.dropout = nn.Dropout(self._full_config.dropout_rate)

        # Rotary embeddings (needed for position_embeddings computation)
        self.rotary_emb = self._init_rotary_emb(encoder_config)

        self.gradient_checkpointing = False

    def _init_rotary_emb(self, encoder_config):
        """Initialize rotary embeddings (copied from T5GemmaEncoder)."""
        from transformers.models.t5gemma.modeling_t5gemma import T5GemmaRotaryEmbedding

        return T5GemmaRotaryEmbedding(encoder_config)

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
        Forward pass with coordinate support for LARA.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            position_ids: Position indices for RoPE
            inputs_embeds: Pre-computed embeddings (batch, seq_len, hidden_size)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            coordinates: 3D coordinates (batch, seq_len, 3) - Passed to each layer

        Returns:
            BaseModelOutput with last_hidden_state, hidden_states, attentions
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Get embeddings
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = self.dropout(inputs_embeds)

        batch_size, seq_length = hidden_states.shape[:2]

        # Prepare position IDs
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Compute position embeddings (RoPE)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Prepare attention mask (expand to 4D if needed)
        if attention_mask is not None and attention_mask.dim() == 2:
            # Expand to 4D: (batch, 1, 1, seq_len)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            # Convert to float and create causal mask
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        # Layer stack
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # Gradient checkpointing
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

        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class GeoT5GemmaDecoderLayer(T5GemmaDecoderLayer):
    """
    T5GemmaDecoderLayer with optional LARA for self-attention and cross-attention.

    Extends the base T5GemmaDecoderLayer to:
    1. Optionally replace self_attn with LARA
    2. Optionally replace cross_attn with CrossLARAAttention
    3. Accept and pass decoder coordinates to attention layers

    This is critical for EDA routing where:
    - Decoder self-attention benefits from spatial path continuity
    - Cross-attention benefits from spatial proximity to source loads
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        enable_geometric_self_attention: bool = False,
        use_geo_cross_attn: bool = False,
        geo_config_dict: Optional[Dict[str, Any]] = None,
    ):
        # Initialize base layer
        super().__init__(config, layer_idx)

        if geo_config_dict is None:
            geo_config_dict = {}

        # Replace self-attention with LARA if enabled
        if enable_geometric_self_attention:
            self.self_attn = T5GemmaLARAAttention(
                config,
                layer_idx=layer_idx,
                coord_scale=geo_config_dict.get('coord_scale', 1e-5),
                use_geometric_bias=geo_config_dict.get('use_geometric_bias', True),
                bias_mlp_hidden=geo_config_dict.get('bias_mlp_hidden', 64),
            )

        # Replace cross-attention with CrossLARAAttention if enabled
        if use_geo_cross_attn:
            self.cross_attn = CrossLARAAttention(
                config,
                layer_idx=layer_idx,
                coord_scale=geo_config_dict.get('coord_scale', 1e-5),
                use_geometric_bias=geo_config_dict.get('use_geometric_bias', True),
                bias_mlp_hidden=geo_config_dict.get('bias_mlp_hidden', 64),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        decoder_coordinates: Optional[torch.Tensor] = None,      # NEW
        encoder_coordinates: Optional[torch.Tensor] = None,      # NEW
        **kwargs
    ):
        """
        Decoder layer forward with coordinate support.

        Args:
            decoder_coordinates: Decoder token coordinates (batch, seq_len, 3)
            encoder_coordinates: Encoder token coordinates (batch, src_len, 3)
        """

        # Self-Attention
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
  
        # Pass coordinates to self-attention if it's LARA
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            coordinates=decoder_coordinates,  # Pass decoder coords
            output_attentions=output_attentions,
            **kwargs
        )
 
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        # Cross-Attention (if encoder_hidden_states provided)
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.pre_cross_attn_layernorm(hidden_states)

            # Check if cross_attn is CrossLARAAttention
            if isinstance(self.cross_attn, CrossLARAAttention):
                # Use CrossLARAAttention with coordinates
                hidden_states, cross_attn_weights, _ = self.cross_attn(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    query_coordinates=decoder_coordinates,    # Decoder coords
                    key_coordinates=encoder_coordinates,      # Encoder coords
                    output_attentions=output_attentions,
                )
            else:
                # Standard cross-attention (returns 2 values, not 3)
                hidden_states, cross_attn_weights = self.cross_attn(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = self.post_cross_attn_layernorm(hidden_states)
            hidden_states = residual + self.dropout(hidden_states)

        # MLP
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states) 
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class GeoT5GemmaDecoder(T5GemmaDecoder):
    """
    T5GemmaDecoder with LARA geometric attention support.

    Extends the base T5GemmaDecoder to:
    1. Use GeoT5GemmaDecoderLayer with optional LARA
    2. Accept and pass decoder coordinates through layer stack
    """

    def __init__(self, config):
        # Call nn.Module.__init__ directly
        nn.Module.__init__(self)

        # Get decoder config
        if hasattr(config, 'decoder'):
            decoder_config = config.decoder
        else:
            decoder_config = config

        # Store decoder config (not full config) - required by parent forward()
        self.config = decoder_config
        # Also store full config for access to geometric_config, vocab_size, etc.
        self._full_config = config

        # Embedding layer
        self.embed_tokens = nn.Embedding(
            self._full_config.vocab_size,
            decoder_config.hidden_size,
            padding_idx=self._full_config.pad_token_id,
        )

        # Create custom decoder layers with optional LARA
        geo_config_dict = getattr(self._full_config, 'geometric_config', {})
        enable_geo_self_attn = geo_config_dict.get('use_geo_self_attn', False)
        enable_geo_cross_attn = geo_config_dict.get('use_geo_cross_attn', False)

        self.layers = nn.ModuleList([
            GeoT5GemmaDecoderLayer(
                decoder_config,
                layer_idx=layer_idx,
                enable_geometric_self_attention=enable_geo_self_attn,
                use_geo_cross_attn=enable_geo_cross_attn,
                geo_config_dict=geo_config_dict,
            )
            for layer_idx in range(decoder_config.num_hidden_layers)
        ])

        self.final_layer_norm = T5GemmaRMSNorm(
            decoder_config.hidden_size,
            eps=getattr(decoder_config, 'rms_norm_eps', 1e-6)
        )

        self.dropout = nn.Dropout(self._full_config.dropout_rate)

        # Rotary embeddings
        self.rotary_emb = self._init_rotary_emb(decoder_config)

        self.gradient_checkpointing = False

    def _init_rotary_emb(self, decoder_config):
        """Initialize rotary embeddings."""
        from transformers.models.t5gemma.modeling_t5gemma import T5GemmaRotaryEmbedding
        return T5GemmaRotaryEmbedding(decoder_config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        decoder_coordinates: Optional[torch.Tensor] = None,       # NEW
        encoder_coordinates: Optional[torch.Tensor] = None,       # NEW
        **kwargs
    ):
        """
        Decoder forward with coordinate support.

        Args:
            decoder_coordinates: (batch, seq_len, 3)
            encoder_coordinates: (batch, src_len, 3)
            **kwargs: Additional arguments (may contain coordinates from parent calls)
        """

        # Extract coordinates from kwargs if not provided directly
        if decoder_coordinates is None and 'decoder_coordinates' in kwargs:
            decoder_coordinates = kwargs.pop('decoder_coordinates')
        if encoder_coordinates is None and 'encoder_coordinates' in kwargs:
            encoder_coordinates = kwargs.pop('encoder_coordinates')
        # Also check for encoder_abs_positions (alternative name)
        if encoder_coordinates is None and 'encoder_abs_positions' in kwargs:
            encoder_coordinates = kwargs.pop('encoder_abs_positions')
        """
        Decoder forward with coordinate support.

        Args:
            decoder_coordinates: (batch, seq_len, 3)
            encoder_coordinates: (batch, src_len, 3)
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Get embeddings
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = self.dropout(inputs_embeds)

        batch_size, seq_length = hidden_states.shape[:2]

        # Prepare position IDs
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Compute position embeddings (RoPE)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Prepare attention masks
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        # Layer stack
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions and encoder_hidden_states is not None else None
        next_decoder_cache = () if use_cache else None

        # Handle different cache formats (tuple vs Cache object)
        # In newer transformers versions, past_key_values can be a Cache object
        # Check if it's an empty Cache object and treat it as None
        if past_key_values is not None:
            if hasattr(past_key_values, '__len__') and not isinstance(past_key_values, (tuple, list)):
                # It's a Cache object - check if it has any cached layers
                if len(past_key_values) == 0:
                    # Empty cache, treat as None
                    past_key_values = None

        for idx, decoder_layer in enumerate[Module](self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Get past_key_value for this layer
            # Use try/except to safely handle different cache formats
            past_key_value = None
            if past_key_values is not None:
                try:
                    past_key_value = past_key_values[idx]
                except (KeyError, IndexError, TypeError):
                    past_key_value = None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    decoder_coordinates,   # Pass coords
                    encoder_coordinates,   # Pass coords
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    decoder_coordinates=decoder_coordinates,   # Pass coords
                    encoder_coordinates=encoder_coordinates,   # Pass coords
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Return with past_key_values for caching support
        next_cache = next_decoder_cache if use_cache else None

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class GeoT5GemmaForConditionalGeneration(T5GemmaForConditionalGeneration):
    """
    T5Gemma with Geometry-Aware Position Embeddings for EDA routing.

    This model extends the base T5GemmaForConditionalGeneration with:

    1. **Geometry-Aware Position Embedding (Recommended)**:
       - Only semantic tokens (<DRIVER>, <LOAD>) receive position embeddings
       - XY coordinates: Fourier encoding for continuous 2D positions
       - Metal layer: Learnable embedding with direction awareness
       - Relative position: Polar + Circular Harmonics for (Δx, Δy)
       - Layer delta: Signed embedding for via traversal direction

    2. **Simple Fourier Position Embedding (Alternative)**:
       - Maps all 3D coordinates (x, y, z) uniformly to sinusoidal features
       - Less sophisticated but simpler

    3. **Geometric Attention (Step 3, optional)**:
       - LARA (Lie Algebra Relative Attention) for geometry-aware attention
       - Combines semantic similarity with geometric displacement

    Position Embedding Design:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Token Type        │  Position Embedding                                 │
    │  ──────────────────┼───────────────────────────────────────────────────  │
    │  <DRIVER>          │  Abs_PE(driver_x, driver_y, driver_m)               │
    │  <LOAD_i>          │  Abs_PE(load_i) + Rel_PE(load_i - driver)           │
    │  Other tokens      │  Zero (no position embedding)                       │
    └─────────────────────────────────────────────────────────────────────────┘

    The model accepts additional inputs:
    - encoder_abs_positions: Absolute 3D coordinates (batch, src_len, 3)
    - encoder_rel_positions: Relative 3D coordinates (batch, src_len, 3)
    - decoder_coordinates: 3D coordinates for decoder tokens (batch, tgt_len, 3)

    Args:
        config: T5GemmaConfig for the base model
        geo_config: GeoConfig or dict with geometric embedding settings

    Example:
        >>> from transformers import T5GemmaConfig
        >>> config = T5GemmaConfig(hidden_size=256, num_hidden_layers=4)
        >>> geo_config = GeoConfig(use_advanced_geo_pe=True)
        >>> model = GeoT5GemmaForConditionalGeneration(config, geo_config)
        >>>
        >>> # Forward pass with coordinates
        >>> outputs = model(
        ...     input_ids=input_ids,
        ...     attention_mask=attention_mask,
        ...     encoder_abs_positions=encoder_abs_pos,  # (batch, src_len, 3)
        ...     encoder_rel_positions=encoder_rel_pos,  # (batch, src_len, 3)
        ...     labels=labels,
        ...     decoder_coordinates=decoder_coords,     # (batch, tgt_len, 3)
        ... )
    """

    def __init__(
        self,
        config: T5GemmaConfig,
        geo_config: Optional[Union[GeoConfig, Dict[str, Any]]] = None
    ):
        # Parse geo_config first (before calling super().__init__)
        if geo_config is None:
            self.geo_config = GeoConfig()
        elif isinstance(geo_config, dict):
            self.geo_config = GeoConfig.from_dict(geo_config)
        else:
            self.geo_config = geo_config

        # Store geometric config in model config for layer access
        config.geometric_config = self.geo_config.to_dict()

        # Initialize base model
        super().__init__(config)

        # Replace encoder/decoder with Geo versions if LARA is enabled
        # Note: T5GemmaForConditionalGeneration stores encoder/decoder in self.model
        if self.geo_config.use_geo_self_attn:
            # Encoder: only use LARA if explicitly enabled for encoder
            # (by default we don't use LARA in encoder due to sparse coordinates)
            enable_encoder_lara = config.geometric_config.get('enable_encoder_lara', False)
            if enable_encoder_lara:
                self.model.encoder = GeoT5GemmaEncoder(config)

            # Decoder: always use LARA when geometric_attention is enabled
            # This is the most important part for EDA routing
            self.model.decoder = GeoT5GemmaDecoder(config)

        # Initialize Position Embedding modules
        self.encoder_geo_pe = None
        self.decoder_geo_pe = None
        self.encoder_fourier_pe = None
        self.decoder_fourier_pe = None

        # Option 1: Geometry-Aware Position Embedding (Recommended)
        # Uses separate encodings for XY (Fourier), Metal Layer (Learnable),
        # Relative Position (Polar + Harmonics), and Layer Delta (Signed)
        if self.geo_config.use_advanced_geo_pe:
            geope_config = self.geo_config.to_geope_config(config.hidden_size)
            self.encoder_geo_pe = GeometryAwarePositionEmbedding(geope_config)
            # Decoder uses simple Fourier since it only has cumulative positions
            self.decoder_fourier_pe = FourierPositionEmbedding(
                hidden_size=config.hidden_size,
                num_frequencies=self.geo_config.num_frequencies,
                max_wavelength=self.geo_config.max_wavelength,
                min_wavelength=self.geo_config.min_wavelength,
                coord_scale=self.geo_config.coord_scale,
                learnable_coefficients=self.geo_config.learnable_fourier_coefficients,
                separate_basis=self.geo_config.separate_sin_cos_basis,
                floor_freq_ratio=self.geo_config.floor_freq_ratio,
                max_sequence_length=self.geo_config.max_sequence_length,
            )

        # Option 1.5: Metal Layer Only Position Embedding (For Testing)
        # Only uses Metal Layer encoding to test its isolated effect
        elif self.geo_config.use_metal_layer_only_pe:
            geope_config = self.geo_config.to_geope_config(config.hidden_size)
            self.encoder_geo_pe = GeometryAwarePositionEmbeddingTMP(geope_config)
            # Decoder uses simple Fourier since it only has cumulative positions
            self.decoder_fourier_pe = FourierPositionEmbedding(
                hidden_size=config.hidden_size,
                num_frequencies=self.geo_config.num_frequencies,
                max_wavelength=self.geo_config.max_wavelength,
                min_wavelength=self.geo_config.min_wavelength,
                coord_scale=self.geo_config.coord_scale,
                learnable_coefficients=self.geo_config.learnable_fourier_coefficients,
                separate_basis=self.geo_config.separate_sin_cos_basis,
                floor_freq_ratio=self.geo_config.floor_freq_ratio,
                max_sequence_length=self.geo_config.max_sequence_length,
            )

        # Option 2: Simple Fourier Position Embedding (Alternative)
        elif self.geo_config.use_basic_fourier_pe:
            self.encoder_fourier_pe = FourierPositionEmbedding(
                hidden_size=config.hidden_size,
                num_frequencies=self.geo_config.num_frequencies,
                max_wavelength=self.geo_config.max_wavelength,
                min_wavelength=self.geo_config.min_wavelength,
                coord_scale=self.geo_config.coord_scale,
                learnable_coefficients=self.geo_config.learnable_fourier_coefficients,
                separate_basis=self.geo_config.separate_sin_cos_basis,
                floor_freq_ratio=self.geo_config.floor_freq_ratio,
                max_sequence_length=self.geo_config.max_sequence_length,
            )
            self.decoder_fourier_pe = FourierPositionEmbedding(
                hidden_size=config.hidden_size,
                num_frequencies=self.geo_config.num_frequencies,
                max_wavelength=self.geo_config.max_wavelength,
                min_wavelength=self.geo_config.min_wavelength,
                coord_scale=self.geo_config.coord_scale,
                learnable_coefficients=self.geo_config.learnable_fourier_coefficients,
                separate_basis=self.geo_config.separate_sin_cos_basis,
                floor_freq_ratio=self.geo_config.floor_freq_ratio,
                max_sequence_length=self.geo_config.max_sequence_length,
            )

        # Vector Quantization module (information bottleneck for PE)
        self.encoder_pe_vq = None
        if self.geo_config.use_vq and (
            self.encoder_geo_pe is not None or self.encoder_fourier_pe is not None
        ):
            from .vq import VectorQuantizer
            self.encoder_pe_vq = VectorQuantizer(
                hidden_size=config.hidden_size,
                codebook_size=self.geo_config.vq_codebook_size,
                commitment_cost=self.geo_config.vq_commitment_cost,
                ema_decay=self.geo_config.vq_ema_decay,
                dead_code_threshold=self.geo_config.vq_dead_code_threshold,
            )

        # Initialize VQ loss accumulator
        self._vq_loss = None

        # Post-initialization
        self.post_init()

    def _add_encoder_geometric_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        abs_positions: Optional[torch.Tensor],
        rel_positions: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Add geometry-aware position embeddings to encoder input embeddings.

        This method handles two cases:
        1. Geometry-Aware PE: Uses abs_positions + rel_positions with separate
           encodings for XY, metal layer, relative polar, and layer delta.
        2. Simple Fourier PE: Uses only abs_positions with uniform Fourier encoding.

        Position Embedding Design:
        - <DRIVER> token: abs_pos = driver position, rel_pos = (0, 0, 0)
        - <LOAD> tokens: abs_pos = load position, rel_pos = (load - driver)
        - Other tokens: abs_pos = (0, 0, 0), rel_pos = (0, 0, 0) → Zero PE

        Args:
            inputs_embeds: Token embeddings (batch, seq_len, hidden_size)
            abs_positions: Absolute 3D coordinates (batch, seq_len, 3)
            rel_positions: Relative 3D coordinates (batch, seq_len, 3)
            attention_mask: Optional attention mask

        Returns:
            Enhanced embeddings with geometric information
        """
        # Case 1: Geometry-Aware Position Embedding (Recommended)
        if self.encoder_geo_pe is not None:
            if abs_positions is None or rel_positions is None:
                return inputs_embeds
            # GeometryAwarePositionEmbedding expects (abs_pos, rel_pos)
            # Note: Module handles dtype internally for fp16 compatibility
            geo_embeds = self.encoder_geo_pe(
                abs_positions,
                rel_positions,
                attention_mask
            )
            # Ensure dtype matches inputs_embeds (critical for fp16 training)
            if geo_embeds.dtype != inputs_embeds.dtype:
                geo_embeds = geo_embeds.to(inputs_embeds.dtype)

            # Apply VQ if enabled (information bottleneck)
            if self.encoder_pe_vq is not None:
                geo_embeds, vq_loss, _ = self.encoder_pe_vq(geo_embeds, attention_mask)
                self._vq_loss = vq_loss  # Store for adding to main loss

            return inputs_embeds + geo_embeds

        # Case 2: Simple Fourier Position Embedding
        if self.encoder_fourier_pe is not None and abs_positions is not None:
            # Note: Module handles dtype internally for fp16 compatibility
            geo_embeds = self.encoder_fourier_pe(abs_positions, attention_mask)
            # Ensure dtype matches inputs_embeds (critical for fp16 training)
            if geo_embeds.dtype != inputs_embeds.dtype:
                geo_embeds = geo_embeds.to(inputs_embeds.dtype)

            # Apply VQ if enabled (information bottleneck)
            if self.encoder_pe_vq is not None:
                geo_embeds, vq_loss, _ = self.encoder_pe_vq(geo_embeds, attention_mask)
                self._vq_loss = vq_loss  # Store for adding to main loss

            return inputs_embeds + geo_embeds

        return inputs_embeds

    def _add_decoder_geometric_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        coordinates: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decoder geometric embeddings are intentionally disabled.

        In autoregressive EDA routing, the decoder predicts the next token
        given previous tokens. Adding Fourier PE from ground-truth absolute
        coordinates (decoder_coordinates) would leak label information into
        the decoder input, causing data leakage.

        The decoder relies solely on token embeddings + cross-attention to
        the geometry-aware encoder representations.

        Args:
            inputs_embeds: Token embeddings (batch, seq_len, hidden_size)
            coordinates: 3D coordinates (unused, kept for interface compat)
            attention_mask: Optional attention mask (unused)

        Returns:
            Original input embeddings without geometric modification
        """
        return inputs_embeds

    def _shift_decoder_coordinates(
        self,
        decoder_coordinates: torch.Tensor,
        encoder_abs_positions: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Shift decoder_coordinates right by 1 to prevent data leakage.

        Problem: tgt_coords[t] is the position AFTER executing target_tokens[t],
        which is aligned with labels[t]. During autoregressive decoding at position t,
        the model should not see the coordinate resulting from labels[t] — that would
        leak the answer (coords[t] - coords[t-1] directly reveals labels[t]).

        Solution: Shift right so that decoder_coordinates[t] = position BEFORE
        predicting labels[t], i.e., the position after all previously decoded tokens.

        Before shift (data leakage):
            target_tokens:      T2,     <PUSH>, D30000, D3000, ...
            tgt_coords:         [A],    [B],    [C],    [D],   ...
            decoder_input:      <pad>,  T2,     <PUSH>, D30000, ...
            decoder_coordinates:[A],    [B],    [C],    [D],   ...  ← coords[t] leaks labels[t]

        After shift (correct):
            decoder_input:      <pad>,  T2,     <PUSH>, D30000, ...
            decoder_coordinates:[drv],  [A],    [B],    [C],   ...  ← coords[t] = current position
        """
        # Work in FP32 to prevent overflow (chip coords can be >65504, FP16 max)
        # Extract driver starting position from encoder_abs_positions
        # Source format: "<BOS> <DRIVER> ..." → driver is at index 1
        if encoder_abs_positions is not None:
            driver_pos = encoder_abs_positions[:, 1:2, :].float()  # (batch, 1, 3)
        else:
            driver_pos = torch.zeros(
                decoder_coordinates.shape[0], 1, 3,
                dtype=torch.float32, device=decoder_coordinates.device
            )


        # Shift right: prepend driver starting position, drop last coordinate
        # Keep in FP32 — downstream LARA modules call .float() internally
        shifted = torch.cat([driver_pos, decoder_coordinates[:, :-1, :].float()], dim=1)
        return shifted

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # Geometry-Aware Position Embedding inputs (Recommended)
        encoder_abs_positions: Optional[torch.Tensor] = None,
        encoder_rel_positions: Optional[torch.Tensor] = None,
        decoder_coordinates: Optional[torch.Tensor] = None,
        # Legacy input (for backward compatibility with simple Fourier PE)
        encoder_coordinates: Optional[torch.Tensor] = None,
        # Accept additional kwargs for compatibility with newer transformers versions
        # (e.g., cache_position added in transformers >= 4.43.0)
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        """
        Forward pass with geometry-aware position embeddings.

        This extends the base T5Gemma forward pass by:
        1. Computing token embeddings from input_ids
        2. Adding position embeddings based on coordinates:
           - Geometry-Aware PE: Uses abs_positions + rel_positions
           - Simple Fourier PE: Uses encoder_coordinates only
        3. Passing enhanced embeddings to encoder/decoder

        Position Embedding Design (Geometry-Aware PE):
        ┌─────────────────────────────────────────────────────────────────────┐
        │  Token Type        │  Position Embedding                            │
        │  ──────────────────┼──────────────────────────────────────────────  │
        │  <DRIVER>          │  GeoPE(abs_pos, rel_pos=(0,0,0))               │
        │  <LOAD_i>          │  GeoPE(abs_pos, rel_pos=(load - driver))       │
        │  Other tokens      │  Zero PE (abs_pos=(0,0,0), rel_pos=(0,0,0))    │
        └─────────────────────────────────────────────────────────────────────┘

        New Args:
            encoder_abs_positions: Absolute 3D positions (batch, src_len, 3)
                - <DRIVER>: driver's (x, y, m)
                - <LOAD>: load's (x, y, m)
                - Others: (0, 0, 0)
            encoder_rel_positions: Relative 3D positions (batch, src_len, 3)
                - <DRIVER>: (0, 0, 0)
                - <LOAD>: (load - driver) = (Δx, Δy, Δm)
                - Others: (0, 0, 0)
            decoder_coordinates: Cumulative 3D positions for decoder (batch, tgt_len, 3)
            encoder_coordinates: Legacy input for simple Fourier PE (batch, src_len, 3)

        Returns:
            Seq2SeqLMOutput with loss, logits, and other outputs
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Handle legacy encoder_coordinates input
        # If encoder_abs_positions not provided but encoder_coordinates is,
        # use encoder_coordinates as abs_positions (backward compatibility)
        if encoder_abs_positions is None and encoder_coordinates is not None:
            encoder_abs_positions = encoder_coordinates

        # Get token embeddings if not provided
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Add position embeddings to encoder inputs
        if inputs_embeds is not None:
            inputs_embeds = self._add_encoder_geometric_embeddings(
                inputs_embeds,
                encoder_abs_positions,
                encoder_rel_positions,
                attention_mask,
            )

        # Get decoder token embeddings if needed
        if decoder_inputs_embeds is None and decoder_input_ids is not None:
            decoder_inputs_embeds = self.get_input_embeddings()(decoder_input_ids)

        # Add position embeddings to decoder inputs
        if decoder_inputs_embeds is not None:
            decoder_inputs_embeds = self._add_decoder_geometric_embeddings(
                decoder_inputs_embeds,
                decoder_coordinates,
                decoder_attention_mask,
            )

        # === KEY CHANGE: Handle encoder call with coordinates ===
        # If encoder_outputs not cached, call encoder explicitly
        encoder = self.get_encoder()
        if encoder_outputs is None:
            if isinstance(encoder, GeoT5GemmaEncoder):
                # GeoT5GemmaEncoder - pass coordinates for LARA
                encoder_outputs = encoder(
                    input_ids=None,  # Already have inputs_embeds
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    coordinates=encoder_abs_positions,  # Pass coordinates to LARA layers
                )
            else:
                # Standard encoder
                encoder_outputs = encoder(
                    input_ids=None,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
        
        # === KEY FIX: Call decoder explicitly if using GeoT5GemmaDecoder ===
        # The base class forward doesn't pass decoder_coordinates through,
        # so we need to handle the decoder call ourselves
        decoder = self.get_decoder()
        if isinstance(decoder, GeoT5GemmaDecoder):
            # Prepare decoder input ids (shifted labels for training)
            if labels is not None and decoder_inputs_embeds is None:
                decoder_input_ids = self._shift_right(labels)
                decoder_inputs_embeds = self.get_input_embeddings()(decoder_input_ids)
                # Apply decoder position embeddings (no-op in current impl)
                decoder_inputs_embeds = self._add_decoder_geometric_embeddings(
                    decoder_inputs_embeds,
                    decoder_coordinates,
                    decoder_attention_mask,
                ) 

            # Shift decoder_coordinates to prevent data leakage during training.
            # tgt_coords[t] = position after target_tokens[t] (aligned with labels[t]).
            # After shift: decoder_coordinates[t] = position before predicting labels[t].
            if decoder_coordinates is not None and labels is not None:
                decoder_coordinates = self._shift_decoder_coordinates(
                    decoder_coordinates, encoder_abs_positions
                )

            # Prepare encoder attention mask
            if attention_mask is not None and attention_mask.dim() == 2:
                # Use encoder outputs dtype as reference since decoder_inputs_embeds may be None in edge cases
                dtype = encoder_outputs[0].dtype if hasattr(encoder_outputs[0], 'dtype') else inputs_embeds.dtype
                encoder_attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                encoder_attention_mask = encoder_attention_mask.to(dtype=dtype)
                encoder_attention_mask = (1.0 - encoder_attention_mask) * torch.finfo(dtype).min
            else:
                encoder_attention_mask = attention_mask

            # Call GeoT5GemmaDecoder with coordinates
            decoder_outputs = decoder(
                input_ids=None,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                decoder_coordinates=decoder_coordinates,  # Pass shifted decoder coords to LARA
                encoder_coordinates=encoder_abs_positions,  # Pass encoder coords for cross-LARA
            )
 
            # Compute logits and loss
            lm_logits = self.lm_head(decoder_outputs[0])

            loss = None
            if labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    lm_logits.view(-1, lm_logits.size(-1)),
                    labels.view(-1)
                )

                # Add VQ loss if applicable
                if self._vq_loss is not None:
                    loss = loss + self._vq_loss
                    self._vq_loss = None  # Reset

            if not return_dict:
                output = (lm_logits,) + decoder_outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return Seq2SeqLMOutput(
                loss=loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )

        # Fallback: Standard decoder (no LARA) - use parent forward
        output = super().forward(
            input_ids=None,  # Use inputs_embeds instead
            attention_mask=attention_mask,
            decoder_input_ids=None,  # Use decoder_inputs_embeds instead
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,  # Pass through additional args like cache_position
        )

        # Add VQ loss if applicable
        if self._vq_loss is not None:
            if return_dict and output.loss is not None:
                # Modify the dataclass output
                output.loss = output.loss + self._vq_loss
            elif not return_dict and isinstance(output, tuple) and len(output) > 0:
                # Modify tuple output (loss is first element if present)
                if isinstance(output[0], torch.Tensor) and output[0].dim() == 0:
                    output = (output[0] + self._vq_loss,) + output[1:]
            self._vq_loss = None  # Reset

        return output

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        """
        Prepare inputs for generation, handling geometric coordinates.

        This extends the base prepare_inputs_for_generation to:
        1. Pass through encoder coordinate information
        2. Track decoder coordinates in real-time using CoordinateTracker

        Coordinate tracking strategy:
        - For training: decoder_coordinates provided as input
        - For inference: Use _coordinate_tracker to parse generated tokens
                        and update coordinates incrementally

        Coordinate inputs handled:
        - encoder_abs_positions: Absolute positions for encoder (static)
        - encoder_rel_positions: Relative positions for encoder (static)
        - decoder_coordinates: Cumulative positions for decoder (dynamic)
        - _coordinate_tracker: InferenceCoordinateTracker instance (inference only)
        """
        # Get base preparation
        model_inputs = super().prepare_inputs_for_generation(
            decoder_input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            encoder_outputs=encoder_outputs,
            **kwargs
        )

        # Pass through encoder coordinate information (static during generation)
        if "encoder_abs_positions" in kwargs:
            model_inputs["encoder_abs_positions"] = kwargs["encoder_abs_positions"]
        if "encoder_rel_positions" in kwargs:
            model_inputs["encoder_rel_positions"] = kwargs["encoder_rel_positions"]
        if "encoder_coordinates" in kwargs:
            model_inputs["encoder_coordinates"] = kwargs["encoder_coordinates"]

        # Handle decoder coordinates
        if "decoder_coordinates" in kwargs:
            # Explicit decoder_coordinates provided (training mode)
            model_inputs["decoder_coordinates"] = kwargs["decoder_coordinates"]
        elif hasattr(self, '_coordinate_tracker') and self._coordinate_tracker is not None:
            # Inference mode with coordinate tracker
            # Parse the last generated token and update tracker
            if decoder_input_ids.shape[1] > self._last_decoder_length:
                # New token was generated
                last_token_id = decoder_input_ids[0, -1].item()
                last_token_str = self._tokenizer.decode(
                    [last_token_id], skip_special_tokens=False
                )
                # Update tracker with new token
                self._coordinate_tracker.update(last_token_str)
                self._last_decoder_length = decoder_input_ids.shape[1]

            # Get current coordinates from tracker
            decoder_coords = self._coordinate_tracker.get_batch_coords_tensor(
                batch_size=decoder_input_ids.shape[0],
                device=decoder_input_ids.device
            )
            model_inputs["decoder_coordinates"] = decoder_coords

        return model_inputs

    def generate(
        self,
        input_ids=None,
        encoder_abs_positions=None,
        encoder_rel_positions=None,
        driver_pos=(0, 0, 0),
        use_coordinate_tracking=True,
        **kwargs
    ):
        """
        Generate with optional real-time coordinate tracking.

        Args:
            input_ids: Encoder input token IDs
            encoder_abs_positions: Encoder absolute positions
            encoder_rel_positions: Encoder relative positions
            driver_pos: Starting position for decoder (driver coordinate)
            use_coordinate_tracking: Whether to use real-time coordinate tracking
            **kwargs: Additional arguments for parent generate()

        Returns:
            Generated token IDs

        Example:
            >>> outputs = model.generate(
            ...     input_ids=input_ids,
            ...     encoder_abs_positions=encoder_abs_pos,
            ...     driver_pos=(2382120, 691600, 2),
            ...     max_length=512,
            ... )
        """
        # Check if LARA is enabled for decoder
        decoder = self.get_decoder()
        if use_coordinate_tracking and isinstance(decoder, GeoT5GemmaDecoder):
            # Initialize coordinate tracker for inference
            from .coordinate_tracker import InferenceCoordinateTracker

            self._coordinate_tracker = InferenceCoordinateTracker(driver_pos=driver_pos)
            self._last_decoder_length = 1  # BOS token

            # Store tokenizer for token decoding
            if not hasattr(self, '_tokenizer'):
                raise ValueError(
                    "Tokenizer not found. Please set model._tokenizer before calling generate()."
                )
        else:
            self._coordinate_tracker = None

        # Add encoder positions to kwargs
        if encoder_abs_positions is not None:
            kwargs['encoder_abs_positions'] = encoder_abs_positions
        if encoder_rel_positions is not None:
            kwargs['encoder_rel_positions'] = encoder_rel_positions

        try:
            # Call parent generate
            outputs = super().generate(input_ids=input_ids, **kwargs)
            return outputs
        finally:
            # Clean up tracker
            if hasattr(self, '_coordinate_tracker'):
                delattr(self, '_coordinate_tracker')
            if hasattr(self, '_last_decoder_length'):
                delattr(self, '_last_decoder_length')

    def _reorder_cache(
        self, past_key_values, beam_idx: torch.Tensor
    ):
        """
        Reorder past key values for beam search.

        This method is required for beam search to work properly. It reorders
        the cached key-value pairs according to the beam indices selected at
        each generation step.

        Args:
            past_key_values: The cached key-value pairs from previous steps.
                Can be a tuple of tuples or a Cache object.
            beam_idx: Tensor of beam indices to reorder by.

        Returns:
            Reordered past_key_values in the same format as input.
        """
        # Handle None case
        if past_key_values is None:
            return None

        # Handle Cache object (newer transformers versions)
        if hasattr(past_key_values, 'reorder_cache'):
            # Cache object has its own reorder method
            past_key_values.reorder_cache(beam_idx)
            return past_key_values

        # Handle legacy tuple format
        reordered_past = ()
        for layer_past in past_key_values:
            if layer_past is None:
                reordered_past = reordered_past + (None,)
            else:
                # Each layer_past is a tuple of (key, value) tensors
                # Reorder along the batch dimension (dim 0)
                reordered_layer = tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                )
                reordered_past = reordered_past + (reordered_layer,)

        return reordered_past

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        geo_config: Optional[Union[GeoConfig, Dict[str, Any]]] = None,
        **kwargs
    ) -> "GeoT5GemmaForConditionalGeneration":
        """
        Load pretrained GeoT5GemmaForConditionalGeneration model.

        This method properly handles the geometric_config that was saved
        with the model. If geo_config is not provided, it will be loaded
        from the saved config.json.

        Args:
            pretrained_model_name_or_path: Path to pretrained model or checkpoint
            geo_config: Optional geometric config (if None, loads from saved config)
            **kwargs: Additional arguments for from_pretrained

        Returns:
            GeoT5GemmaForConditionalGeneration with all weights loaded
        """
        import json
        from pathlib import Path

        # Load config first to get geometric_config
        config = T5GemmaConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Try to get geometric_config from saved config
        saved_geo_config = getattr(config, 'geometric_config', None)

        # Use provided geo_config, or fall back to saved, or default
        if geo_config is not None:
            # User provided geo_config
            if isinstance(geo_config, dict):
                final_geo_config = GeoConfig.from_dict(geo_config)
            else:
                final_geo_config = geo_config
        elif saved_geo_config is not None:
            # Load from saved config
            if isinstance(saved_geo_config, dict):
                final_geo_config = GeoConfig.from_dict(saved_geo_config)
            else:
                final_geo_config = saved_geo_config
        else:
            # Default config
            final_geo_config = GeoConfig()

        # Store in config for model initialization
        config.geometric_config = final_geo_config.to_dict()

        # Create model with the geo config
        model = cls.__new__(cls)

        # Initialize geo_config before calling parent __init__
        model.geo_config = final_geo_config

        # Call parent __init__ which sets up the base model
        T5GemmaForConditionalGeneration.__init__(model, config)

        # Replace encoder/decoder with Geo versions if LARA is enabled
        if model.geo_config.use_geo_self_attn:
            enable_encoder_lara = config.geometric_config.get('enable_encoder_lara', False)
            if enable_encoder_lara:
                model.model.encoder = GeoT5GemmaEncoder(config)
            model.model.decoder = GeoT5GemmaDecoder(config)

        # Initialize Position Embedding modules
        model.encoder_geo_pe = None
        model.decoder_geo_pe = None
        model.encoder_fourier_pe = None
        model.decoder_fourier_pe = None

        if model.geo_config.use_advanced_geo_pe:
            geope_config = model.geo_config.to_geope_config(config.hidden_size)
            model.encoder_geo_pe = GeometryAwarePositionEmbedding(geope_config)
            model.decoder_fourier_pe = FourierPositionEmbedding(
                hidden_size=config.hidden_size,
                num_frequencies=model.geo_config.num_frequencies,
                max_wavelength=model.geo_config.max_wavelength,
                min_wavelength=model.geo_config.min_wavelength,
                coord_scale=model.geo_config.coord_scale,
                learnable_coefficients=model.geo_config.learnable_fourier_coefficients,
                separate_basis=model.geo_config.separate_sin_cos_basis,
                floor_freq_ratio=model.geo_config.floor_freq_ratio,
                max_sequence_length=model.geo_config.max_sequence_length,
            )
        elif model.geo_config.use_metal_layer_only_pe:
            geope_config = model.geo_config.to_geope_config(config.hidden_size)
            model.encoder_geo_pe = GeometryAwarePositionEmbeddingTMP(geope_config)
            model.decoder_fourier_pe = FourierPositionEmbedding(
                hidden_size=config.hidden_size,
                num_frequencies=model.geo_config.num_frequencies,
                max_wavelength=model.geo_config.max_wavelength,
                min_wavelength=model.geo_config.min_wavelength,
                coord_scale=model.geo_config.coord_scale,
                learnable_coefficients=model.geo_config.learnable_fourier_coefficients,
                separate_basis=model.geo_config.separate_sin_cos_basis,
                floor_freq_ratio=model.geo_config.floor_freq_ratio,
                max_sequence_length=model.geo_config.max_sequence_length,
            )
        elif model.geo_config.use_basic_fourier_pe:
            model.encoder_fourier_pe = FourierPositionEmbedding(
                hidden_size=config.hidden_size,
                num_frequencies=model.geo_config.num_frequencies,
                max_wavelength=model.geo_config.max_wavelength,
                min_wavelength=model.geo_config.min_wavelength,
                coord_scale=model.geo_config.coord_scale,
                learnable_coefficients=model.geo_config.learnable_fourier_coefficients,
                separate_basis=model.geo_config.separate_sin_cos_basis,
                floor_freq_ratio=model.geo_config.floor_freq_ratio,
                max_sequence_length=model.geo_config.max_sequence_length,
            )
            model.decoder_fourier_pe = FourierPositionEmbedding(
                hidden_size=config.hidden_size,
                num_frequencies=model.geo_config.num_frequencies,
                max_wavelength=model.geo_config.max_wavelength,
                min_wavelength=model.geo_config.min_wavelength,
                coord_scale=model.geo_config.coord_scale,
                learnable_coefficients=model.geo_config.learnable_fourier_coefficients,
                separate_basis=model.geo_config.separate_sin_cos_basis,
                floor_freq_ratio=model.geo_config.floor_freq_ratio,
                max_sequence_length=model.geo_config.max_sequence_length,
            )

        # Vector Quantization module (information bottleneck for PE)
        model.encoder_pe_vq = None
        if model.geo_config.use_vq and (
            model.encoder_geo_pe is not None or model.encoder_fourier_pe is not None
        ):
            from .vq import VectorQuantizer
            model.encoder_pe_vq = VectorQuantizer(
                hidden_size=config.hidden_size,
                codebook_size=model.geo_config.vq_codebook_size,
                commitment_cost=model.geo_config.vq_commitment_cost,
                ema_decay=model.geo_config.vq_ema_decay,
                dead_code_threshold=model.geo_config.vq_dead_code_threshold,
            )

        # Initialize VQ loss accumulator
        model._vq_loss = None

        # Now load the state dict with all weights
        model_path = Path(pretrained_model_name_or_path)
        if (model_path / "model.safetensors").exists():
            from safetensors.torch import load_file
            state_dict = load_file(model_path / "model.safetensors")
        elif (model_path / "pytorch_model.bin").exists():
            import torch
            state_dict = torch.load(model_path / "pytorch_model.bin", map_location="cpu")
        else:
            # Try loading with parent method for other formats
            state_dict = None

        if state_dict is not None:
            # Load all weights (including geo_pe weights)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                import logging
                logging.warning(f"Missing keys when loading model: {missing_keys}")
            if unexpected_keys:
                import logging
                logging.warning(f"Unexpected keys when loading model: {unexpected_keys}")

        return model

    @classmethod
    def from_pretrained_with_geo(
        cls,
        pretrained_model_name_or_path: str,
        geo_config: Optional[Union[GeoConfig, Dict[str, Any]]] = None,
        **kwargs
    ) -> "GeoT5GemmaForConditionalGeneration":
        """
        Load pretrained T5Gemma and add geometric embeddings.

        DEPRECATED: Use from_pretrained() instead, which now handles geo_config properly.

        This method loads a pretrained T5Gemma model and adds the geometric
        embedding components (Fourier PE) on top.

        Args:
            pretrained_model_name_or_path: Path to pretrained model
            geo_config: Geometric embedding configuration
            **kwargs: Additional arguments for from_pretrained

        Returns:
            GeoT5GemmaForConditionalGeneration with pretrained weights
        """
        import warnings
        warnings.warn(
            "from_pretrained_with_geo is deprecated. Use from_pretrained() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Load base model config
        config = T5GemmaConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Create model with geo config
        model = cls(config, geo_config)

        # Load pretrained weights (only for base model components)
        base_model = T5GemmaForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        # Copy base model weights
        model.load_state_dict(base_model.state_dict(), strict=False)

        return model


def create_geo_t5gemma_config(
    vocab_size: int,
    hidden_size: int = 256,
    intermediate_size: int = 1024,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    head_dim: int = 64,
    max_position_embeddings: int = 512,
    sliding_window: int = 256,
    dropout_rate: float = 0.1,
    bos_token_id: int = 1,
    eos_token_id: int = 2,
    pad_token_id: int = 0,
) -> T5GemmaConfig:
    """
    Create T5GemmaConfig for GeoT5Gemma model.

    This is a convenience function matching the configuration used in
    TrainingPipeline._initialize_T5Gemma_model().

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension
        intermediate_size: FFN intermediate dimension
        num_hidden_layers: Number of encoder/decoder layers
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of key-value heads (for GQA)
        head_dim: Dimension per attention head
        max_position_embeddings: Maximum sequence length
        sliding_window: Sliding window size for attention
        dropout_rate: Dropout probability
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID
        pad_token_id: Padding token ID

    Returns:
        T5GemmaConfig ready for model initialization
    """
    encoder_config = T5GemmaModuleConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        sliding_window=sliding_window,
        tie_word_embeddings=True,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        decoder_start_token_id=bos_token_id,
        attn_implementation="eager",
        use_cache=True,
    )

    decoder_config = T5GemmaModuleConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        sliding_window=sliding_window,
        tie_word_embeddings=True,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        decoder_start_token_id=bos_token_id,
        attn_implementation="eager",
        use_cache=True,
    )

    config = T5GemmaConfig(
        encoder=encoder_config,
        decoder=decoder_config,
        hidden_size=hidden_size,
        query_pre_attn_scalar=hidden_size,
        is_encoder_decoder=True,
        dropout_rate=dropout_rate,
        tie_word_embeddings=True,
        vocab_size=vocab_size,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        decoder_start_token_id=bos_token_id,
        attn_implementation="eager",
        use_cache=True,
    )

    return config
