#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   coordinate_vq.py
@Time    :   2026/03/15
@Author  :   Junfeng Liu
@Version :   1.0
@Desc    :   Coordinate-level Vector Quantization for spatial generalization.

             Quantizes 3D coordinates to a learned codebook of spatial prototypes,
             creating an information bottleneck that forces similar positions to
             share identical representations throughout the model.

             Key insight: VQ should be applied at the COORDINATE level (3D),
             not at the embedding level (256D). This ensures all downstream
             computations (Fourier PE, LARA quaternion rotation, geometric bias)
             work with quantized coordinates, providing consistent spatial
             coarsening across all components.

             Why this improves generalization:
             - Raw Fourier features: sin(ω·1000) ≠ sin(ω·1001) for high ω
               → nearby positions get DIFFERENT representations → fragile
             - VQ coordinates: VQ(1000) = VQ(1001) if in same Voronoi cell
               → nearby positions get IDENTICAL representations → robust

             Used at 3 integration points:
             ① Encoder PE: abs_pos, rel_pos → VQ → Fourier features → MLP
             ② LARA self-attention: coordinates → VQ → GeoPE + GeoBias
             ③ CrossLARA: query/key coordinates → VQ → GeoPE + GeoBias

             Reference:
             - "Neural Discrete Representation Learning" (VQ-VAE, NeurIPS 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CoordinateVQ(nn.Module):
    """
    Vector Quantization for 3D spatial coordinates.

    Operates in per-axis-scaled coordinate space: raw coordinates are first
    scaled by (coord_scale_xy, coord_scale_xy, coord_scale_z), then quantized
    to the nearest codebook entry, then scaled back to original units.

    The codebook lives in normalized space where:
    - x, y axes: range ~[0, 50] (after 1e-5 scaling of chip coords ~0-5M)
    - z axis: range ~[0, 6] (after 0.3 scaling of metal layers 0-20)

    Information capacity: log2(codebook_size) bits per token position.
    - K=128: 7 bits, ~5 bins/axis → coarse
    - K=256: 8 bits, ~6 bins/axis → moderate (recommended start)
    - K=512: 9 bits, ~8 bins/axis → finer

    Args:
        codebook_size: Number of codebook entries (K)
        coord_dim: Input coordinate dimension (3 for x,y,z)
        coord_scale_xy: Scale factor for x,y axes
        coord_scale_z: Scale factor for z axis
        commitment_cost: β for commitment loss
        ema_decay: EMA decay for codebook updates
        dead_code_threshold: Usage threshold for dead code revival

    Example:
        >>> vq = CoordinateVQ(codebook_size=256)
        >>> coords = torch.tensor([[[1000000, 2000000, 3],
        ...                         [1000500, 2000200, 3]]], dtype=torch.float)
        >>> quantized, loss, indices = vq(coords)
        >>> # If 1000000 and 1000500 fall in same Voronoi cell after scaling,
        >>> # quantized[0,0] == quantized[0,1] → identical downstream features
    """

    def __init__(
        self,
        codebook_size: int = 256,
        coord_dim: int = 3,
        coord_scale_xy: float = 1e-5,
        coord_scale_z: float = 0.3,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        dead_code_threshold: int = 2,
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.coord_dim = coord_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold

        # Per-axis scaling to normalize coordinate ranges before VQ.
        # This ensures L2 distance treats all axes fairly.
        self.register_buffer(
            'coord_scale',
            torch.tensor([coord_scale_xy, coord_scale_xy, coord_scale_z])
        )

        # Codebook in normalized coordinate space.
        # No gradient: updated via EMA.
        self.codebook = nn.Embedding(codebook_size, coord_dim)
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.codebook.weight.requires_grad_(False)

        # EMA tracking buffers
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_embedding_sum', self.codebook.weight.data.clone())
        self.register_buffer('initialized', torch.tensor(False))
        self.register_buffer('usage_count', torch.zeros(codebook_size))

    @torch.no_grad()
    def _initialize_codebook(self, flat_inputs: torch.Tensor):
        """Initialize codebook from first batch of valid scaled coordinates."""
        n = flat_inputs.shape[0]
        if n >= self.codebook_size:
            indices = torch.randperm(n, device=flat_inputs.device)[:self.codebook_size]
            self.codebook.weight.data.copy_(flat_inputs[indices])
        else:
            repeats = (self.codebook_size + n - 1) // n
            expanded = flat_inputs.repeat(repeats, 1)[:self.codebook_size]
            noise = torch.randn_like(expanded) * 0.01
            self.codebook.weight.data.copy_(expanded + noise)

        self.ema_cluster_size.fill_(1.0)
        self.ema_embedding_sum.copy_(self.codebook.weight.data)
        self.initialized.fill_(True)

    @torch.no_grad()
    def _revive_dead_codes(self, flat_inputs: torch.Tensor):
        """Reinitialize codebook entries with low usage."""
        dead_mask = self.usage_count < self.dead_code_threshold
        n_dead = dead_mask.sum().item()
        if n_dead == 0 or flat_inputs.shape[0] == 0:
            return

        n = flat_inputs.shape[0]
        indices = torch.randint(0, n, (n_dead,), device=flat_inputs.device)
        noise = torch.randn(n_dead, self.coord_dim, device=flat_inputs.device) * 0.01
        new_codes = (flat_inputs[indices] + noise).to(self.codebook.weight.dtype)
        self.codebook.weight.data[dead_mask] = new_codes
        self.ema_cluster_size[dead_mask] = torch.ones(
            n_dead, device=self.ema_cluster_size.device, dtype=self.ema_cluster_size.dtype
        )
        self.ema_embedding_sum[dead_mask] = new_codes.to(self.ema_embedding_sum.dtype)
        self.usage_count[dead_mask] = torch.full(
            (n_dead,), self.dead_code_threshold,
            device=self.usage_count.device, dtype=self.usage_count.dtype
        )

    def forward(
        self,
        coordinates: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize 3D coordinates via nearest-neighbor codebook lookup.

        Flow: raw coords → scale → VQ in normalized space → unscale → output

        Args:
            coordinates: Raw coordinates (batch, seq_len, 3)
            attention_mask: Optional mask (batch, seq_len), 1=valid, 0=pad

        Returns:
            quantized_coords: Quantized in original scale (batch, seq_len, 3)
            vq_loss: Scalar commitment loss
            indices: Codebook indices (batch, seq_len)
        """
        batch_size, seq_len, D = coordinates.shape
        input_dtype = coordinates.dtype
        device = coordinates.device

        device_type = 'cuda' if coordinates.is_cuda else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            # Scale to normalized space (FP32)
            coords_scaled = coordinates.float() * self.coord_scale  # (B, T, 3)

            # Identify valid tokens: non-padding AND has real coordinates
            has_coords = (coordinates.abs().sum(dim=-1) > 0)  # (B, T)
            if attention_mask is not None:
                valid = has_coords & attention_mask.bool()
            else:
                valid = has_coords
            valid_flat = valid.reshape(-1)  # (B*T,)

            flat_scaled = coords_scaled.reshape(-1, D)  # (B*T, 3)
            flat_valid = flat_scaled[valid_flat]  # (N_valid, 3)

            if flat_valid.shape[0] == 0:
                zero_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                return coordinates, zero_loss, torch.zeros(
                    batch_size, seq_len, dtype=torch.long, device=device
                )

            # Lazy initialization from first batch
            if self.training and not self.initialized:
                self._initialize_codebook(flat_valid.detach())

            # Nearest codebook entry: ||z - e||² = ||z||² - 2<z,e> + ||e||²
            cb = self.codebook.weight.float()  # (K, 3)
            distances = (
                flat_valid.pow(2).sum(dim=-1, keepdim=True)
                - 2 * flat_valid @ cb.t()
                + cb.pow(2).sum(dim=-1, keepdim=True).t()
            )  # (N_valid, K)

            indices_valid = distances.argmin(dim=-1)  # (N_valid,)
            quantized_valid = F.embedding(indices_valid, cb)  # (N_valid, 3)

            # Commitment loss: encoder output → codebook
            commitment_loss = F.mse_loss(flat_valid, quantized_valid.detach())
            vq_loss = self.commitment_cost * commitment_loss

            # EMA codebook update (training only)
            if self.training:
                with torch.no_grad():
                    encodings = F.one_hot(indices_valid, self.codebook_size).float()
                    cluster_size = encodings.sum(dim=0)

                    self.ema_cluster_size.mul_(self.ema_decay).add_(
                        cluster_size, alpha=1 - self.ema_decay
                    )
                    embedding_sum = encodings.t() @ flat_valid
                    self.ema_embedding_sum.mul_(self.ema_decay).add_(
                        embedding_sum, alpha=1 - self.ema_decay
                    )

                    # Laplace smoothing
                    n = self.ema_cluster_size.sum()
                    cluster_size_smooth = (
                        (self.ema_cluster_size + 1e-5)
                        / (n + self.codebook_size * 1e-5) * n
                    )
                    self.codebook.weight.data.copy_(
                        self.ema_embedding_sum / cluster_size_smooth.unsqueeze(-1)
                    )

                    self.usage_count.mul_(self.ema_decay).add_(
                        (cluster_size > 0).float(), alpha=1 - self.ema_decay
                    )
                    self._revive_dead_codes(flat_valid)

            # Straight-through estimator
            quantized_valid = flat_valid + (quantized_valid - flat_valid).detach()

            # Scatter back to full tensor
            quantized_flat = flat_scaled.clone()  # Start from scaled coords
            quantized_flat[valid_flat] = quantized_valid

            indices_flat = torch.zeros(batch_size * seq_len, dtype=torch.long, device=device)
            indices_flat[valid_flat] = indices_valid

            # Unscale back to original coordinate space
            quantized_coords = quantized_flat.reshape(batch_size, seq_len, D) / self.coord_scale

        return quantized_coords.to(input_dtype), vq_loss, indices_flat.reshape(batch_size, seq_len)

    def extra_repr(self) -> str:
        return (
            f"codebook_size={self.codebook_size}, "
            f"coord_dim={self.coord_dim}, "
            f"commitment_cost={self.commitment_cost}, "
            f"ema_decay={self.ema_decay}"
        )
