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
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

from .position_embedding import (
    FourierPositionEmbedding,
    GeometryAwarePositionEmbedding,
    GeoPEConfig,
)
from .geometric_attention import LieAlgebraRelativeAttention, T5GemmaLARAAttention


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
        enable_fourier_pe: Whether to use simple Fourier Position Embedding
        enable_geometry_aware_pe: Whether to use advanced Geometry-Aware PE (overrides enable_fourier_pe)
        enable_geometric_attention: Whether to use LARA geometric attention
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
    enable_fourier_pe: bool = False  # Simple 3D Fourier (deprecated)
    enable_geometry_aware_pe: bool = True  # Advanced Geometry-Aware PE (recommended)
    enable_geometric_attention: bool = False  # Start with just PE, no LARA
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
        enable_geometric_attention: Whether to use LARA instead of standard attention
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        enable_geometric_attention: bool = False,
        geo_config_dict: Optional[Dict[str, Any]] = None,
    ):
        # Initialize base layer (creates standard self_attn)
        super().__init__(config, layer_idx)

        # Replace self-attention with LARA if enabled
        if enable_geometric_attention:
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

        self.config = config

        # Get encoder config (handle both T5GemmaConfig and T5GemmaModuleConfig)
        if hasattr(config, 'encoder'):
            encoder_config = config.encoder
        else:
            encoder_config = config

        # Embedding layer
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            encoder_config.hidden_size,
            padding_idx=config.pad_token_id,
        )

        # Create custom encoder layers with optional LARA support
        geo_config_dict = getattr(config, 'geometric_config', {})
        enable_geo_attn = geo_config_dict.get('enable_geometric_attention', False)

        # Pass encoder_config to layers (not the full config)
        self.layers = nn.ModuleList([
            GeoT5GemmaEncoderLayer(
                encoder_config,  # Pass encoder_config, not full config
                layer_idx=layer_idx,
                enable_geometric_attention=enable_geo_attn,
                geo_config_dict=geo_config_dict,  # Pass geo_config_dict
            )
            for layer_idx in range(encoder_config.num_hidden_layers)
        ])

        self.final_layer_norm = T5GemmaRMSNorm(
            encoder_config.hidden_size,
            eps=getattr(encoder_config, 'rms_norm_eps', 1e-6)
        )

        self.dropout = nn.Dropout(config.dropout_rate)

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
        >>> geo_config = GeoConfig(enable_geometry_aware_pe=True)
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

        # Replace encoder with GeoT5GemmaEncoder if LARA is enabled
        if self.geo_config.enable_geometric_attention:
            self.encoder = GeoT5GemmaEncoder(config)
            # Note: Decoder can also be replaced similarly if needed
            # For now, we only replace encoder as decoder is autoregressive
            # and may not benefit as much from geometric attention

        # Initialize Position Embedding modules
        self.encoder_geo_pe = None
        self.decoder_geo_pe = None
        self.encoder_fourier_pe = None
        self.decoder_fourier_pe = None

        # Option 1: Geometry-Aware Position Embedding (Recommended)
        # Uses separate encodings for XY (Fourier), Metal Layer (Learnable),
        # Relative Position (Polar + Harmonics), and Layer Delta (Signed)
        if self.geo_config.enable_geometry_aware_pe:
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

        # Option 2: Simple Fourier Position Embedding (Alternative)
        elif self.geo_config.enable_fourier_pe:
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
            geo_embeds = self.encoder_geo_pe(
                abs_positions.float(),
                rel_positions.float(),
                attention_mask
            )
            return inputs_embeds + geo_embeds

        # Case 2: Simple Fourier Position Embedding
        if self.encoder_fourier_pe is not None and abs_positions is not None:
            geo_embeds = self.encoder_fourier_pe(abs_positions.float(), attention_mask)
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
        # If using GeoT5GemmaEncoder and encoder_outputs not cached,
        # call encoder explicitly to pass coordinates
        if encoder_outputs is None and isinstance(self.encoder, GeoT5GemmaEncoder):
            encoder_outputs = self.encoder(
                input_ids=None,  # Already have inputs_embeds
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                coordinates=encoder_abs_positions,  # Pass coordinates to LARA layers
            )

        # Call parent forward with enhanced embeddings and encoder outputs
        return super().forward(
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
        )

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

        This extends the base prepare_inputs_for_generation to pass through
        coordinate information during autoregressive generation.

        Coordinate inputs handled:
        - encoder_abs_positions: Absolute positions for encoder (new)
        - encoder_rel_positions: Relative positions for encoder (new)
        - decoder_coordinates: Cumulative positions for decoder
        - encoder_coordinates: Legacy input for simple Fourier PE
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

        # Pass through coordinate information (Geometry-Aware PE)
        if "encoder_abs_positions" in kwargs:
            model_inputs["encoder_abs_positions"] = kwargs["encoder_abs_positions"]
        if "encoder_rel_positions" in kwargs:
            model_inputs["encoder_rel_positions"] = kwargs["encoder_rel_positions"]
        if "decoder_coordinates" in kwargs:
            model_inputs["decoder_coordinates"] = kwargs["decoder_coordinates"]

        # Legacy support for simple Fourier PE
        if "encoder_coordinates" in kwargs:
            model_inputs["encoder_coordinates"] = kwargs["encoder_coordinates"]

        return model_inputs

    @classmethod
    def from_pretrained_with_geo(
        cls,
        pretrained_model_name_or_path: str,
        geo_config: Optional[Union[GeoConfig, Dict[str, Any]]] = None,
        **kwargs
    ) -> "GeoT5GemmaForConditionalGeneration":
        """
        Load pretrained T5Gemma and add geometric embeddings.

        This method loads a pretrained T5Gemma model and adds the geometric
        embedding components (Fourier PE) on top.

        Args:
            pretrained_model_name_or_path: Path to pretrained model
            geo_config: Geometric embedding configuration
            **kwargs: Additional arguments for from_pretrained

        Returns:
            GeoT5GemmaForConditionalGeneration with pretrained weights
        """
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
