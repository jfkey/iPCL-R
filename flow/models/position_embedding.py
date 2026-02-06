#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   position_embedding.py
@Time    :   2025/01/14
@Author  :   Dawn Li
@Version :   2.0
@Desc    :   Geometry-Aware Position Embedding for EDA Routing.

             This module implements a sophisticated position embedding scheme that
             captures both absolute and relative spatial relationships in chip routing:

             1. XY Absolute Position (Fourier): Multi-frequency encoding for (x, y)
             2. Metal Layer Encoding (Learnable): Layer-aware embedding with direction
             3. XY Relative Position (Polar + Circular Harmonics): Distance + direction
             4. Layer Relative Position (Signed Embedding): Via traversal direction

             Mathematical Foundation:
             - Fourier PE: PE(p) = [sin(ω_i p), cos(ω_i p)] for multi-scale representation
             - Circular Harmonics: [sin(kθ), cos(kθ)] for rotation-aware direction encoding
             - Polar Decomposition: (Δx, Δy) → (r, θ) for magnitude-direction separation

             Reference:
             - "Fourier Position Embedding" (ICML 2024)
             - "Fourier Features Let Networks Learn High Frequency Functions" (NeurIPS 2020)
             - "GeoPE: A Unified Geometric Positional Embedding" (Arxiv 2025)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierPositionEmbedding(nn.Module):
    """
    Fourier Position Embedding for 3D chip coordinates.

    This module maps absolute 3D coordinates (x, y, z) to high-frequency
    sinusoidal features that are added to token embeddings. This enables
    the model to perceive geometric positions in the chip routing space.

    Key features:
    - Multi-frequency encoding with logarithmically spaced frequency bands
    - Optional learnable Fourier coefficients for adaptability
    - Frequency clipping to zero-out undertrained low frequencies
    - Separate sin/cos basis for enhanced expressivity

    The encoding follows the Fourier feature formula:
        γ(p) = [sin(2π σ₁ p), cos(2π σ₁ p), ..., sin(2π σ_L p), cos(2π σ_L p)]

    For 3D coordinates:
        FourierPE(x, y, z) = Concat[γ(x), γ(y), γ(z)] · W_proj

    Args:
        hidden_size: Output dimension (should match model hidden_size)
        num_frequencies: Number of frequency bands per coordinate axis
        max_wavelength: Maximum wavelength for frequency bands
        min_wavelength: Minimum wavelength for frequency bands
        coord_scale: Scaling factor for input coordinates
        learnable_coefficients: Whether to use learnable Fourier mixing coefficients
        separate_basis: Whether to use separate coefficients for sin/cos
        floor_freq_ratio: Ratio for clipping low frequencies (vs 2π/max_seq_len)
        max_sequence_length: Maximum sequence length for frequency clipping

    Example:
        >>> pe = FourierPositionEmbedding(hidden_size=256, num_frequencies=64)
        >>> coords = torch.randn(2, 100, 3) * 10000  # (batch, seq_len, 3)
        >>> embeddings = pe(coords)  # (batch, seq_len, 256)
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_frequencies: int = 64,
        max_wavelength: float = 10000.0,
        min_wavelength: float = 1.0,
        coord_scale: float = 1e-4,
        learnable_coefficients: bool = True,
        separate_basis: bool = True,
        floor_freq_ratio: float = 1.0,
        max_sequence_length: int = 512,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_frequencies = num_frequencies
        self.coord_scale = coord_scale
        self.learnable_coefficients = learnable_coefficients
        self.separate_basis = separate_basis

        # Frequency bands (logarithmically spaced)
        # Higher frequencies capture fine-grained spatial details
        # Lower frequencies capture global structure
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0 / max_wavelength),
                math.log(1.0 / min_wavelength),
                num_frequencies
            )
        ) * 2 * math.pi

        # Zero-out undertrained frequencies (FoPE insight)
        # Frequencies below 2π/max_seq_len are undertrained and can cause
        # spectral damage during length generalization
        floor_freq = 2 * math.pi / max_sequence_length * floor_freq_ratio
        freqs = freqs.clone()
        freqs[freqs < floor_freq] = 0.0

        self.register_buffer('frequencies', freqs)  # (num_frequencies,)

        # Input dimension: 3 coords × num_frequencies × 2 (sin/cos)
        input_dim = 3 * num_frequencies * 2

        # Output projection to hidden_size
        self.output_proj = nn.Linear(input_dim, hidden_size)

        # Learnable Fourier coefficients (FoPE enhancement)
        # These allow the model to learn optimal frequency mixing
        if learnable_coefficients:
            if separate_basis:
                # Separate coefficients for sin and cos basis
                # This provides more flexibility in combining phases
                self.sin_coef = nn.Parameter(
                    torch.randn(num_frequencies, num_frequencies) * 0.3
                )
                self.cos_coef = nn.Parameter(
                    torch.randn(num_frequencies, num_frequencies) * 0.3
                )
                # Initialize close to identity for stable training start
                nn.init.eye_(self.sin_coef)
                nn.init.eye_(self.cos_coef)
            else:
                # Shared coefficients for both sin and cos
                self.fourier_coef = nn.Parameter(
                    torch.randn(num_frequencies, num_frequencies) * 0.3
                )
                nn.init.eye_(self.fourier_coef)

    def forward(
        self,
        coordinates: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Fourier position embeddings for 3D coordinates.

        Args:
            coordinates: Tensor of shape (batch, seq_len, 3) with (x, y, z) values
                        Coordinates should be in chip units (potentially large integers)
            attention_mask: Optional mask of shape (batch, seq_len) for padded positions

        Returns:
            Position embeddings of shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = coordinates.shape
        # Store original dtype for fp16 compatibility
        input_dtype = coordinates.dtype
        # Get model dtype from output_proj weights (may be fp16 in mixed precision)
        model_dtype = self.output_proj.weight.dtype

        # Scale coordinates to reasonable range
        # Chip coordinates can be millions; scaling to ~1.0 range helps stability
        # Use model dtype for compatibility with learnable parameters
        coords_scaled = coordinates.to(model_dtype) * self.coord_scale  # (B, T, 3)

        # Separate x, y, z coordinates
        x = coords_scaled[..., 0:1]  # (B, T, 1)
        y = coords_scaled[..., 1:2]  # (B, T, 1)
        z = coords_scaled[..., 2:3]  # (B, T, 1)

        # Compute phase: coord × frequency
        # frequencies: (num_freq,) -> (1, 1, num_freq) for broadcasting
        # Cast buffer to model_dtype for fp16 compatibility
        freqs = self.frequencies.to(model_dtype).view(1, 1, -1)

        phase_x = x * freqs  # (B, T, num_freq)
        phase_y = y * freqs  # (B, T, num_freq)
        phase_z = z * freqs  # (B, T, num_freq)

        # Apply learnable coefficients (Fourier series mixing)
        if self.learnable_coefficients:
            if self.separate_basis:
                # Normalize coefficients to prevent explosion
                sin_coef_norm = self.sin_coef / (
                    self.sin_coef.sum(dim=0, keepdim=True).clamp(min=1e-6)
                )
                cos_coef_norm = self.cos_coef / (
                    self.cos_coef.sum(dim=0, keepdim=True).clamp(min=1e-6)
                )

                # Mix frequencies via learned coefficients
                # This allows the model to combine multiple frequency components
                sin_x = torch.einsum('btf,fg->btg', torch.sin(phase_x), sin_coef_norm)
                cos_x = torch.einsum('btf,fg->btg', torch.cos(phase_x), cos_coef_norm)
                sin_y = torch.einsum('btf,fg->btg', torch.sin(phase_y), sin_coef_norm)
                cos_y = torch.einsum('btf,fg->btg', torch.cos(phase_y), cos_coef_norm)
                sin_z = torch.einsum('btf,fg->btg', torch.sin(phase_z), sin_coef_norm)
                cos_z = torch.einsum('btf,fg->btg', torch.cos(phase_z), cos_coef_norm)
            else:
                coef_norm = self.fourier_coef / (
                    self.fourier_coef.sum(dim=0, keepdim=True).clamp(min=1e-6)
                )
                sin_x = torch.einsum('btf,fg->btg', torch.sin(phase_x), coef_norm)
                cos_x = torch.einsum('btf,fg->btg', torch.cos(phase_x), coef_norm)
                sin_y = torch.einsum('btf,fg->btg', torch.sin(phase_y), coef_norm)
                cos_y = torch.einsum('btf,fg->btg', torch.cos(phase_y), coef_norm)
                sin_z = torch.einsum('btf,fg->btg', torch.sin(phase_z), coef_norm)
                cos_z = torch.einsum('btf,fg->btg', torch.cos(phase_z), coef_norm)
        else:
            # Standard Fourier features without mixing
            sin_x, cos_x = torch.sin(phase_x), torch.cos(phase_x)
            sin_y, cos_y = torch.sin(phase_y), torch.cos(phase_y)
            sin_z, cos_z = torch.sin(phase_z), torch.cos(phase_z)

        # Concatenate all Fourier features
        # Order: [sin_x, cos_x, sin_y, cos_y, sin_z, cos_z]
        fourier_features = torch.cat([
            sin_x, cos_x, sin_y, cos_y, sin_z, cos_z
        ], dim=-1)  # (B, T, 6 * num_freq)

        # Check dtype match before Linear call
        if fourier_features.dtype != self.output_proj.weight.dtype:
            fourier_features = fourier_features.to(self.output_proj.weight.dtype)

        # Project to hidden_size
        position_embeddings = self.output_proj(fourier_features)

        # Mask padded positions if needed
        if attention_mask is not None:
            position_embeddings = position_embeddings * attention_mask.unsqueeze(-1).to(model_dtype)

        # Output in model_dtype (compatible with mixed precision training)
        return position_embeddings

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_frequencies={self.num_frequencies}, "
            f"coord_scale={self.coord_scale}, "
            f"learnable={self.learnable_coefficients}"
        )


# =============================================================================
# Geometry-Aware Position Embedding (Enhanced Design)
# =============================================================================

@dataclass
class GeoPEConfig:
    """Configuration for Geometry-Aware Position Embedding.

    Attributes:
        hidden_size: Output dimension (model hidden size)
        num_frequencies: Number of frequency bands for Fourier encoding
        num_harmonics: Number of circular harmonics for direction encoding
        max_metal_layers: Maximum number of metal layers (for embedding table)
        max_layer_delta: Maximum layer difference for relative encoding
        coord_scale: Scaling factor for large chip coordinates
        max_wavelength: Maximum wavelength for Fourier frequencies
        min_wavelength: Minimum wavelength for Fourier frequencies
        learnable_frequencies: Whether to learn frequency scales
        dropout: Dropout rate for position embeddings
    """
    hidden_size: int = 512
    num_frequencies: int = 32
    num_harmonics: int = 8
    max_metal_layers: int = 16
    max_layer_delta: int = 10
    coord_scale: float = 1e-5
    max_wavelength: float = 10000.0
    min_wavelength: float = 1.0
    learnable_frequencies: bool = True
    dropout: float = 0.1


class XYFourierEmbedding(nn.Module):
    """
    Fourier Position Embedding for 2D (x, y) coordinates.

    Encodes continuous spatial coordinates using multi-frequency sinusoidal functions.
    This captures both local fine-grained positions (high frequency) and global
    spatial structure (low frequency).

    Mathematical formulation:
        PE(x) = [sin(ω_1 x), cos(ω_1 x), sin(ω_2 x), cos(ω_2 x), ...]

    where ω_i = 2π / λ_i and λ_i spans from min_wavelength to max_wavelength.

    Args:
        num_frequencies: Number of frequency bands
        max_wavelength: Maximum wavelength (for lowest frequency)
        min_wavelength: Minimum wavelength (for highest frequency)
        coord_scale: Scaling factor for input coordinates
        learnable: Whether to learn frequency mixing coefficients
    """

    def __init__(
        self,
        num_frequencies: int = 32,
        max_wavelength: float = 10000.0,
        min_wavelength: float = 1.0,
        coord_scale: float = 1e-5,
        learnable: bool = True,
    ):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.coord_scale = coord_scale
        self.output_dim = num_frequencies * 2 * 2  # sin/cos for x and y

        # Logarithmically spaced frequencies
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0 / max_wavelength),
                math.log(1.0 / min_wavelength),
                num_frequencies
            )
        ) * 2 * math.pi

        if learnable:
            self.frequencies = nn.Parameter(freqs)
        else:
            self.register_buffer('frequencies', freqs)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute Fourier embeddings for (x, y) coordinates.

        Args:
            coords: Tensor of shape (..., 2) or (..., 3) with (x, y, [m])

        Returns:
            Fourier embeddings of shape (..., output_dim)
        """
        # Scale coordinates
        x = coords[..., 0:1] * self.coord_scale  # (..., 1)
        y = coords[..., 1:2] * self.coord_scale  # (..., 1)

        # Compute phases: (..., 1) * (num_freq,) -> (..., num_freq)
        # Cast frequencies to input dtype for fp16 compatibility
        freqs = self.frequencies.to(coords.dtype).view(*([1] * (coords.dim() - 1)), -1)
        phase_x = x * freqs
        phase_y = y * freqs

        # Fourier features: [sin_x, cos_x, sin_y, cos_y]
        return torch.cat([
            torch.sin(phase_x), torch.cos(phase_x),
            torch.sin(phase_y), torch.cos(phase_y),
        ], dim=-1)


class MetalLayerEmbedding(nn.Module):
    """
    Learnable embedding for metal layer index with direction awareness.

    Metal layers in EDA have specific properties:
    - Discrete values (1, 2, 3, ..., M)
    - Alternating routing directions (odd=vertical, even=horizontal)
    - Layer-specific resistance and capacitance

    This module learns:
    1. Layer embedding: Unique representation for each layer
    2. Direction embedding: Horizontal vs Vertical layer preference

    Mathematical formulation:
        PE_m(m) = Embed_layer(m) + Embed_direction(m mod 2)

    Args:
        max_layers: Maximum number of metal layers
        embed_dim: Embedding dimension
    """

    def __init__(self, max_layers: int = 16, embed_dim: int = 32):
        super().__init__()
        self.max_layers = max_layers
        self.output_dim = embed_dim

        # Layer embedding: one per metal layer
        self.layer_embed = nn.Embedding(max_layers + 1, embed_dim, padding_idx=0)

        # Direction embedding: horizontal (even) vs vertical (odd)
        self.direction_embed = nn.Embedding(2, embed_dim)

        # Initialize with small values
        nn.init.normal_(self.layer_embed.weight, std=0.02)
        nn.init.normal_(self.direction_embed.weight, std=0.02)

    def forward(self, metal_layer: torch.Tensor) -> torch.Tensor:
        """
        Compute metal layer embeddings.

        Args:
            metal_layer: Tensor of metal layer indices (...,)

        Returns:
            Layer embeddings of shape (..., embed_dim)
        """
        # Clamp to valid range
        m = metal_layer.clamp(0, self.max_layers).long()

        # Layer embedding + direction embedding
        layer_emb = self.layer_embed(m)
        direction_emb = self.direction_embed(m % 2)

        return layer_emb + direction_emb


class PolarRelativeEmbedding(nn.Module):
    """
    Polar-coordinate based relative position embedding with Circular Harmonics.

    For relative positions (Δx, Δy), this module:
    1. Converts to polar coordinates: (r, θ) = (√(Δx²+Δy²), atan2(Δy, Δx))
    2. Encodes distance r with Fourier features (multi-scale)
    3. Encodes direction θ with Circular Harmonics (rotation-aware)

    Mathematical formulation:
        r = √(Δx² + Δy²)
        θ = atan2(Δy, Δx)
        PE_rel(r, θ) = Fourier(r) ⊕ [sin(kθ), cos(kθ)]_{k=1}^K

    Properties:
    - Distance encoding captures "how far" (magnitude)
    - Direction encoding captures "which way" (angle)
    - Circular harmonics provide rotation-equivariance

    Args:
        num_frequencies: Number of frequency bands for distance encoding
        num_harmonics: Number of circular harmonics for direction (K)
        max_wavelength: Maximum wavelength for Fourier frequencies
        coord_scale: Scaling factor for coordinates
    """

    def __init__(
        self,
        num_frequencies: int = 32,
        num_harmonics: int = 8,
        max_wavelength: float = 10000.0,
        coord_scale: float = 1e-5,
    ):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.num_harmonics = num_harmonics
        self.coord_scale = coord_scale

        # Output: Fourier(r) * 2 + Harmonics(θ) * 2
        self.output_dim = num_frequencies * 2 + num_harmonics * 2

        # Frequencies for distance encoding
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0 / max_wavelength),
                math.log(1.0),
                num_frequencies
            )
        ) * 2 * math.pi
        self.register_buffer('frequencies', freqs)

        # Harmonic orders: k = 1, 2, ..., K
        harmonics = torch.arange(1, num_harmonics + 1, dtype=torch.float32)
        self.register_buffer('harmonics', harmonics)

    def forward(self, rel_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute polar-coordinate relative position embeddings.

        Args:
            rel_coords: Tensor of shape (..., 2) or (..., 3) with (Δx, Δy, [Δm])

        Returns:
            Polar embeddings of shape (..., output_dim)
        """
        input_dtype = rel_coords.dtype
        dx = rel_coords[..., 0] * self.coord_scale
        dy = rel_coords[..., 1] * self.coord_scale

        # Polar conversion
        r = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)  # Add eps for stability
        theta = torch.atan2(dy, dx)  # Range: [-π, π]

        # Distance encoding: Fourier features
        # Cast buffers to input dtype for fp16 compatibility
        r_expanded = r.unsqueeze(-1)  # (..., 1)
        freqs = self.frequencies.to(input_dtype).view(*([1] * r.dim()), -1)  # (1, ..., num_freq)
        phase_r = r_expanded * freqs
        distance_features = torch.cat([
            torch.sin(phase_r), torch.cos(phase_r)
        ], dim=-1)

        # Direction encoding: Circular Harmonics
        theta_expanded = theta.unsqueeze(-1)  # (..., 1)
        harmonics = self.harmonics.to(input_dtype).view(*([1] * theta.dim()), -1)  # (1, ..., K)
        phase_theta = theta_expanded * harmonics
        direction_features = torch.cat([
            torch.sin(phase_theta), torch.cos(phase_theta)
        ], dim=-1)

        return torch.cat([distance_features, direction_features], dim=-1)


class LayerDeltaEmbedding(nn.Module):
    """
    Embedding for relative metal layer differences (Δm = m_load - m_driver).

    Layer differences indicate via traversal:
    - Δm > 0: Going up in metal stack
    - Δm < 0: Going down in metal stack
    - Δm = 0: Same layer

    This uses a signed embedding approach:
        PE_Δm(Δm) = Embed(Δm + max_delta)  # Shift to non-negative

    Or equivalently:
        PE_Δm(Δm) = sign(Δm) * Embed(|Δm|) + Embed(sign_index)

    Args:
        max_delta: Maximum absolute layer difference
        embed_dim: Embedding dimension
    """

    def __init__(self, max_delta: int = 10, embed_dim: int = 32):
        super().__init__()
        self.max_delta = max_delta
        self.output_dim = embed_dim

        # Embedding table: covers -max_delta to +max_delta
        # Index 0 = -max_delta, Index max_delta = 0, Index 2*max_delta = +max_delta
        self.delta_embed = nn.Embedding(2 * max_delta + 1, embed_dim)
        nn.init.normal_(self.delta_embed.weight, std=0.02)

    def forward(self, delta_m: torch.Tensor) -> torch.Tensor:
        """
        Compute layer delta embeddings.

        Args:
            delta_m: Tensor of layer differences (...,)

        Returns:
            Delta embeddings of shape (..., embed_dim)
        """
        # Shift to non-negative indices: Δm ∈ [-max, +max] → [0, 2*max]
        idx = (delta_m + self.max_delta).clamp(0, 2 * self.max_delta).long()
        return self.delta_embed(idx)


class GeometryAwarePositionEmbedding(nn.Module):
    """
    Comprehensive Geometry-Aware Position Embedding for EDA Routing.

    This module combines four specialized embeddings to capture the complete
    spatial relationships in chip routing:

    1. XY Absolute Position (Fourier): Where is this point in the chip?
    2. Metal Layer (Learnable): Which layer and what routing direction?
    3. XY Relative Position (Polar): How far and which direction from driver?
    4. Layer Relative (Delta): How many via transitions needed?

    Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Input: src_abs_pos = (x, y, m),  src_rel_pos = (Δx, Δy, Δm)       │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
    │  │ XY Fourier  │  │ Layer Embed │  │ Polar Rel   │  │ Layer Δ   │ │
    │  │   (x, y)    │  │    (m)      │  │  (Δx, Δy)   │  │   (Δm)    │ │
    │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘ │
    │         │                │                │               │        │
    │         └────────────────┴────────────────┴───────────────┘        │
    │                                  │                                  │
    │                          ┌───────┴───────┐                          │
    │                          │   Concat      │                          │
    │                          └───────┬───────┘                          │
    │                                  │                                  │
    │                          ┌───────┴───────┐                          │
    │                          │  MLP Fusion   │                          │
    │                          └───────┬───────┘                          │
    │                                  │                                  │
    │                          ┌───────┴───────┐                          │
    │                          │  LayerNorm    │                          │
    │                          └───────┬───────┘                          │
    │                                  ▼                                  │
    │                     Output: (batch, seq_len, hidden_size)           │
    └─────────────────────────────────────────────────────────────────────┘

    Args:
        config: GeoPEConfig with all hyperparameters
    """

    def __init__(self, config: GeoPEConfig):
        super().__init__()
        self.config = config

        # 1. XY Absolute Position Embedding (Fourier)
        self.xy_fourier = XYFourierEmbedding(
            num_frequencies=config.num_frequencies,
            max_wavelength=config.max_wavelength,
            min_wavelength=config.min_wavelength,
            coord_scale=config.coord_scale,
            learnable=config.learnable_frequencies,
        )

        # 2. Metal Layer Embedding (Learnable + Direction)
        self.layer_embed = MetalLayerEmbedding(
            max_layers=config.max_metal_layers,
            embed_dim=config.num_frequencies,  # Match frequency count for balance
        )

        # 3. XY Relative Position Embedding (Polar + Circular Harmonics)
        self.polar_rel = PolarRelativeEmbedding(
            num_frequencies=config.num_frequencies,
            num_harmonics=config.num_harmonics,
            max_wavelength=config.max_wavelength,
            coord_scale=config.coord_scale,
        )

        # 4. Layer Relative Embedding (Δm)
        self.layer_delta = LayerDeltaEmbedding(
            max_delta=config.max_layer_delta,
            embed_dim=config.num_frequencies,
        )

        # Compute total input dimension
        self.total_input_dim = (
            self.xy_fourier.output_dim +      # XY absolute
            self.layer_embed.output_dim +     # Metal layer
            self.polar_rel.output_dim +       # XY relative (polar)
            self.layer_delta.output_dim       # Layer delta
        )

        # MLP Fusion: Combine all embeddings
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.total_input_dim, config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Dropout(config.dropout),
        )

        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # Initialize MLP weights
        self._init_weights()

    def _init_weights(self):
        """Initialize MLP weights with small values for stable start."""
        for module in self.fusion_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        abs_pos: torch.Tensor,
        rel_pos: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute geometry-aware position embeddings.

        Args:
            abs_pos: Absolute positions (batch, seq_len, 3) as (x, y, m)
            rel_pos: Relative positions (batch, seq_len, 3) as (Δx, Δy, Δm)
            attention_mask: Optional mask (batch, seq_len) for padding

        Returns:
            Position embeddings (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = abs_pos.shape
        # Get model dtype from fusion_mlp weights (may be fp16 in mixed precision)
        model_dtype = self.fusion_mlp[0].weight.dtype

        # Convert inputs to model dtype for compatibility with learnable parameters
        abs_pos = abs_pos.to(model_dtype)
        rel_pos = rel_pos.to(model_dtype)

        # 1. XY Absolute Fourier Embedding
        xy_abs_emb = self.xy_fourier(abs_pos)  # (B, T, d_xy)

        # 2. Metal Layer Embedding
        metal_layer = abs_pos[..., 2]  # (B, T)
        layer_emb = self.layer_embed(metal_layer)  # (B, T, d_m)

        # 3. XY Relative Polar Embedding
        polar_rel_emb = self.polar_rel(rel_pos)  # (B, T, d_rel)

        # 4. Layer Delta Embedding
        delta_m = rel_pos[..., 2]  # (B, T)
        layer_delta_emb = self.layer_delta(delta_m)  # (B, T, d_Δm)

        # Concatenate all embeddings (ensure same dtype)
        combined = torch.cat([
            xy_abs_emb.to(model_dtype),
            layer_emb.to(model_dtype),
            polar_rel_emb.to(model_dtype),
            layer_delta_emb.to(model_dtype),
        ], dim=-1)  # (B, T, total_input_dim)

        # Check dtype match before MLP call
        if combined.dtype != self.fusion_mlp[0].weight.dtype:
            combined = combined.to(self.fusion_mlp[0].weight.dtype)

        # MLP Fusion
        fused = self.fusion_mlp(combined)  # (B, T, hidden_size)

        # Layer Normalization
        output = self.layer_norm(fused)

        # Apply attention mask (zero out padding positions)
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1).to(model_dtype)

        return output

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"hidden_size={self.config.hidden_size}, "
            f"num_frequencies={self.config.num_frequencies}, "
            f"num_harmonics={self.config.num_harmonics}, "
            f"total_input_dim={self.total_input_dim}"
        )
