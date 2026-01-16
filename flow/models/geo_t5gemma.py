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
from transformers.modeling_outputs import Seq2SeqLMOutput

from .position_embedding import FourierPositionEmbedding
from .geometric_attention import LieAlgebraRelativeAttention


@dataclass
class GeoConfig:
    """
    Configuration for geometry-aware embeddings.

    This configuration controls all geometric embedding components:
    - Coordinate handling and scaling
    - Fourier Position Embedding parameters
    - Geometric Attention (LARA) parameters

    Attributes:
        enable_fourier_pe: Whether to use Fourier Position Embedding
        enable_geometric_attention: Whether to use LARA geometric attention
        coord_scale: Scaling factor for coordinates (chip coords are large)
        num_frequencies: Number of frequency bands for Fourier embedding
        max_wavelength: Maximum wavelength for frequency bands
        min_wavelength: Minimum wavelength for frequency bands
        learnable_fourier_coefficients: Whether Fourier coefficients are learnable
        separate_sin_cos_basis: Whether to use separate sin/cos coefficient matrices
        floor_freq_ratio: Ratio for clipping low frequencies
        max_sequence_length: Maximum sequence length for frequency clipping
        use_geometric_bias: Whether to use MLP bias in LARA
        bias_mlp_hidden: Hidden dimension for geometric bias MLP
    """

    # General settings
    enable_fourier_pe: bool = True
    enable_geometric_attention: bool = False  # Start with just Fourier PE
    coord_scale: float = 1e-4

    # Fourier Position Embedding settings (Step 2)
    num_frequencies: int = 64
    max_wavelength: float = 10000.0
    min_wavelength: float = 1.0
    learnable_fourier_coefficients: bool = True
    separate_sin_cos_basis: bool = True
    floor_freq_ratio: float = 1.0
    max_sequence_length: int = 512

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


class GeoT5GemmaForConditionalGeneration(T5GemmaForConditionalGeneration):
    """
    T5Gemma with Geometry-Aware Position Embeddings for EDA routing.

    This model extends the base T5GemmaForConditionalGeneration with:

    1. **Fourier Position Embedding (Step 2)**:
       - Maps 3D chip coordinates (x, y, z) to sinusoidal features
       - Added to token embeddings before encoder/decoder processing
       - Enables the model to perceive absolute positions in chip space

    2. **Geometric Attention (Step 3, optional)**:
       - LARA (Lie Algebra Relative Attention) for geometry-aware attention
       - Combines semantic similarity with geometric displacement
       - Can replace standard attention in encoder/decoder layers

    The model accepts additional inputs:
    - encoder_coordinates: 3D coordinates for encoder input tokens
    - decoder_coordinates: 3D coordinates for decoder input tokens

    Args:
        config: T5GemmaConfig for the base model
        geo_config: GeoConfig or dict with geometric embedding settings

    Example:
        >>> from transformers import T5GemmaConfig
        >>> config = T5GemmaConfig(hidden_size=256, num_hidden_layers=4)
        >>> geo_config = GeoConfig(enable_fourier_pe=True)
        >>> model = GeoT5GemmaForConditionalGeneration(config, geo_config)
        >>>
        >>> # Forward pass with coordinates
        >>> outputs = model(
        ...     input_ids=input_ids,
        ...     attention_mask=attention_mask,
        ...     encoder_coordinates=encoder_coords,  # (batch, src_len, 3)
        ...     labels=labels,
        ...     decoder_coordinates=decoder_coords,  # (batch, tgt_len, 3)
        ... )
    """

    def __init__(
        self,
        config: T5GemmaConfig,
        geo_config: Optional[Union[GeoConfig, Dict[str, Any]]] = None
    ):
        super().__init__(config)

        # Parse geo_config
        if geo_config is None:
            self.geo_config = GeoConfig()
        elif isinstance(geo_config, dict):
            self.geo_config = GeoConfig.from_dict(geo_config)
        else:
            self.geo_config = geo_config

        # Initialize Fourier Position Embedding (Step 2)
        if self.geo_config.enable_fourier_pe:
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
        else:
            self.encoder_fourier_pe = None
            self.decoder_fourier_pe = None

        # Note: Geometric Attention (Step 3) integration would require
        # modifying the attention layers. For now, we provide LARA as a
        # standalone module that can be used to replace attention layers
        # if enable_geometric_attention is True.

        # Post-initialization
        self.post_init()

    def _add_geometric_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        coordinates: Optional[torch.Tensor],
        fourier_pe: Optional[FourierPositionEmbedding],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Add geometric (Fourier) position embeddings to input embeddings.

        Args:
            inputs_embeds: Token embeddings (batch, seq_len, hidden_size)
            coordinates: 3D coordinates (batch, seq_len, 3)
            fourier_pe: Fourier position embedding module
            attention_mask: Optional attention mask

        Returns:
            Enhanced embeddings with geometric information
        """
        if fourier_pe is None or coordinates is None:
            return inputs_embeds

        # Compute Fourier position embeddings
        geo_embeds = fourier_pe(coordinates, attention_mask)

        # Add to input embeddings
        return inputs_embeds + geo_embeds

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
        # New geometric inputs
        encoder_coordinates: Optional[torch.Tensor] = None,
        decoder_coordinates: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        """
        Forward pass with geometry-aware position embeddings.

        This extends the base T5Gemma forward pass by:
        1. Computing token embeddings from input_ids
        2. Adding Fourier position embeddings based on coordinates
        3. Passing enhanced embeddings to encoder/decoder

        New Args:
            encoder_coordinates: 3D coordinates for encoder tokens (batch, src_len, 3)
            decoder_coordinates: 3D coordinates for decoder tokens (batch, tgt_len, 3)

        Returns:
            Seq2SeqLMOutput with loss, logits, and other outputs
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get token embeddings if not provided
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.shared(input_ids)

        # Add Fourier position embeddings to encoder inputs
        if inputs_embeds is not None and self.encoder_fourier_pe is not None:
            inputs_embeds = self._add_geometric_embeddings(
                inputs_embeds,
                encoder_coordinates,
                self.encoder_fourier_pe,
                attention_mask,
            )

        # Get decoder token embeddings if needed
        if decoder_inputs_embeds is None and decoder_input_ids is not None:
            decoder_inputs_embeds = self.shared(decoder_input_ids)

        # Add Fourier position embeddings to decoder inputs
        if decoder_inputs_embeds is not None and self.decoder_fourier_pe is not None:
            decoder_inputs_embeds = self._add_geometric_embeddings(
                decoder_inputs_embeds,
                decoder_coordinates,
                self.decoder_fourier_pe,
                decoder_attention_mask,
            )

        # Call parent forward with enhanced embeddings
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

        # Pass through coordinate information
        if "encoder_coordinates" in kwargs:
            model_inputs["encoder_coordinates"] = kwargs["encoder_coordinates"]
        if "decoder_coordinates" in kwargs:
            model_inputs["decoder_coordinates"] = kwargs["decoder_coordinates"]

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
