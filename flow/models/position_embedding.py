#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   position_embedding.py
@Time    :   2025/01/14
@Author  :   Dawn Li
@Version :   1.0
@Desc    :   Fourier Position Embedding for 3D chip coordinates.
             Step 2 of the Geometry-Aware Trajectory Generation upgrade.

             Maps (x, y, z) coordinates to high-frequency sinusoidal features
             that capture spatial relationships in EDA routing.

             Reference:
             - "Fourier Position Embedding" (ICML 2024)
             - "Fourier Features Let Networks Learn High Frequency Functions" (NeurIPS 2020)
"""

import math
from typing import Optional

import torch
import torch.nn as nn


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

        # Scale coordinates to reasonable range
        # Chip coordinates can be millions; scaling to ~1.0 range helps stability
        coords_scaled = coordinates.float() * self.coord_scale  # (B, T, 3)

        # Separate x, y, z coordinates
        x = coords_scaled[..., 0:1]  # (B, T, 1)
        y = coords_scaled[..., 1:2]  # (B, T, 1)
        z = coords_scaled[..., 2:3]  # (B, T, 1)

        # Compute phase: coord × frequency
        # frequencies: (num_freq,) -> (1, 1, num_freq) for broadcasting
        freqs = self.frequencies.view(1, 1, -1)

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

        # Project to hidden_size
        position_embeddings = self.output_proj(fourier_features)

        # Mask padded positions if needed
        if attention_mask is not None:
            position_embeddings = position_embeddings * attention_mask.unsqueeze(-1)

        return position_embeddings

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_frequencies={self.num_frequencies}, "
            f"coord_scale={self.coord_scale}, "
            f"learnable={self.learnable_coefficients}"
        )
