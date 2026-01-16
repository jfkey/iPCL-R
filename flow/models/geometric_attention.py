#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   geometric_attention.py
@Time    :   2025/01/14
@Author  :   Dawn Li
@Version :   1.0
@Desc    :   Geometric Position Embedding and Lie Algebra Relative Attention.
             Step 3 of the Geometry-Aware Trajectory Generation upgrade.

             Implements:
             - GeometricPositionEmbedding: Quaternion-based 3D rotations using
               Lie algebra symmetric averaging
             - LieAlgebraRelativeAttention (LARA): Attention combining semantic
               similarity with geometric displacement

             Reference:
             - "GeoPE: A Unified Geometric Positional Embedding" (Arxiv 2025)
             - "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2023)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricPositionEmbedding(nn.Module):
    """
    Geometric Position Embedding using quaternions and Lie algebra.

    Extends RoPE (Rotary Position Embedding) to 3D space for chip routing
    coordinates. Uses symmetric averaging in Lie algebra to handle the
    non-commutativity of 3D rotations.

    Key insight from GeoPE paper:
    - Standard RoPE uses 2D rotation matrices for each dimension pair
    - For 3D data, we need quaternions (4D representation of 3D rotations)
    - Quaternion multiplication is non-commutative, so order matters
    - Use Lie algebra averaging to create symmetric, order-invariant rotations

    Mathematical formulation:
    For 3D coordinates (x, y, z), we compute three base rotations:
    - r_x = cos(θ_x/2) + sin(θ_x/2)i  (rotation around x-axis)
    - r_y = cos(θ_y/2) + sin(θ_y/2)j  (rotation around y-axis)
    - r_z = cos(θ_z/2) + sin(θ_z/2)k  (rotation around z-axis)

    The symmetric combined rotation uses Lie algebra averaging:
    u = (1/3)(log(r_x) + log(r_y) + log(r_z))  # Average in Lie algebra
    r = exp(u)                                   # Map back to quaternion

    This approach ensures the rotation is independent of axis ordering.

    Args:
        hidden_size: Model hidden size (for reference)
        num_heads: Number of attention heads
        head_dim: Dimension per head (must be divisible by 3 for GeoPE)
        base: Base frequency for position encoding
        coord_scale: Scaling factor for input coordinates

    Example:
        >>> geope = GeometricPositionEmbedding(head_dim=63)  # 63 = 21 × 3
        >>> q = torch.randn(2, 4, 100, 63)  # (batch, heads, seq, head_dim)
        >>> k = torch.randn(2, 4, 100, 63)
        >>> coords = torch.randn(2, 100, 3) * 10000
        >>> q_rot, k_rot = geope(q, k, coords)
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 4,
        head_dim: int = 63,  # Must be divisible by 3
        base: float = 100.0,
        coord_scale: float = 1e-4,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.base = base
        self.coord_scale = coord_scale

        # For 3D GeoPE, we partition head_dim into sub-vectors of size 3
        # Each 3D sub-vector undergoes quaternion rotation
        if head_dim % 3 != 0:
            # Adjust head_dim to be divisible by 3 by padding
            self.effective_head_dim = (head_dim // 3) * 3
            self.needs_padding = True
        else:
            self.effective_head_dim = head_dim
            self.needs_padding = False

        self.num_subvectors = self.effective_head_dim // 3

        # Frequency bands for each sub-vector (similar to RoPE)
        # Lower indices get higher frequencies for finer-grained encoding
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.num_subvectors, dtype=torch.float) / self.num_subvectors)
        )
        self.register_buffer('inv_freq', inv_freq)

    def compute_quaternion(
        self,
        theta_x: torch.Tensor,
        theta_y: torch.Tensor,
        theta_z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute symmetric quaternion via Lie algebra averaging.

        For three axis-aligned rotations r_x, r_y, r_z, we want a single
        combined rotation that is symmetric (order-independent).

        In Lie algebra (tangent space at identity):
        - log(r_x) = (θ_x/2, 0, 0)  (pure quaternion)
        - log(r_y) = (0, θ_y/2, 0)
        - log(r_z) = (0, 0, θ_z/2)

        Average: u = (θ_x/6, θ_y/6, θ_z/6)
        ||u|| = (1/6)√(θ_x² + θ_y² + θ_z²)

        Exponential map back to quaternion:
        exp(u) = cos(||u||) + (u/||u||)sin(||u||)

        Args:
            theta_x: Phase angles for x-axis rotation (B, T, num_subvectors)
            theta_y: Phase angles for y-axis rotation (B, T, num_subvectors)
            theta_z: Phase angles for z-axis rotation (B, T, num_subvectors)

        Returns:
            Tuple of (w, qx, qy, qz) quaternion components, each (B, T, num_subvectors)
        """
        # Compute the magnitude of the Lie algebra vector
        # Θ = (1/3)√(θ_x² + θ_y² + θ_z²)
        # The 1/3 comes from averaging three rotations
        theta_sq_sum = theta_x ** 2 + theta_y ** 2 + theta_z ** 2
        Theta = (1.0 / 3.0) * torch.sqrt(theta_sq_sum + 1e-8)

        half_Theta = Theta / 2.0

        # Quaternion scalar component: w = cos(Θ/2)
        w = torch.cos(half_Theta)

        # Quaternion vector components
        # For unit quaternion: q = cos(Θ/2) + sin(Θ/2)(n_x i + n_y j + n_z k)
        # where n = u/||u|| is the rotation axis
        sin_half_Theta = torch.sin(half_Theta)

        # scale = sin(Θ/2) / (3Θ) handles the normalization
        # As Θ→0, this approaches 1/6 (L'Hopital's rule)
        scale = sin_half_Theta / (3.0 * Theta + 1e-8)

        qx = scale * theta_x
        qy = scale * theta_y
        qz = scale * theta_z

        return w, qx, qy, qz

    def quaternion_rotate(
        self,
        v: torch.Tensor,
        w: torch.Tensor,
        qx: torch.Tensor,
        qy: torch.Tensor,
        qz: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply quaternion rotation to 3D vectors via sandwich product.

        For a quaternion q = w + xi + yj + zk and vector v, the rotation is:
        v' = q v q*  (quaternion sandwich product)

        This is equivalent to R(q) @ v where R is the 3x3 rotation matrix,
        but computing directly with quaternions is more numerically stable.

        Using the formula: v' = v + 2w(q×v) + 2(q×(q×v))
        where q = (qx, qy, qz) is the vector part.

        Args:
            v: Vectors to rotate (..., 3)
            w, qx, qy, qz: Quaternion components

        Returns:
            Rotated vectors (..., 3)
        """
        vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]

        # First cross product: t = q × v
        t0 = qy * vz - qz * vy
        t1 = qz * vx - qx * vz
        t2 = qx * vy - qy * vx

        # v + 2w(q×v)
        v_prime_x = vx + 2.0 * w * t0
        v_prime_y = vy + 2.0 * w * t1
        v_prime_z = vz + 2.0 * w * t2

        # + 2(q×(q×v)) = + 2(q×t)
        v_prime_x = v_prime_x + 2.0 * (qy * t2 - qz * t1)
        v_prime_y = v_prime_y + 2.0 * (qz * t0 - qx * t2)
        v_prime_z = v_prime_z + 2.0 * (qx * t1 - qy * t0)

        return torch.stack([v_prime_x, v_prime_y, v_prime_z], dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        coordinates: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply GeoPE rotation to query and key vectors.

        The rotation encodes absolute 3D positions into the attention mechanism.
        When computing attention scores q_rot · k_rot, the relative position
        information emerges naturally from the rotation difference.

        Args:
            query: Query vectors (batch, num_heads, seq_len, head_dim)
            key: Key vectors (batch, num_heads, seq_len, head_dim)
            coordinates: 3D coordinates (batch, seq_len, 3)

        Returns:
            Tuple of (query_rotated, key_rotated) with same shapes as input
        """
        batch_size, num_heads, seq_len, head_dim = query.shape

        # Scale coordinates to reasonable range
        coords = coordinates.float() * self.coord_scale  # (B, T, 3)
        x_pos = coords[:, :, 0]  # (B, T)
        y_pos = coords[:, :, 1]  # (B, T)
        z_pos = coords[:, :, 2]  # (B, T)

        # Compute phases for each sub-vector
        # theta = position × frequency
        # inv_freq: (num_subvectors,)
        theta_x = torch.einsum('bt,f->btf', x_pos, self.inv_freq)  # (B, T, num_subvectors)
        theta_y = torch.einsum('bt,f->btf', y_pos, self.inv_freq)
        theta_z = torch.einsum('bt,f->btf', z_pos, self.inv_freq)

        # Compute symmetric quaternion for each position
        w, qx, qy, qz = self.compute_quaternion(theta_x, theta_y, theta_z)
        # Each has shape (B, T, num_subvectors)

        # Handle head_dim not divisible by 3
        if self.needs_padding:
            query_to_rotate = query[..., :self.effective_head_dim]
            key_to_rotate = key[..., :self.effective_head_dim]
            query_remainder = query[..., self.effective_head_dim:]
            key_remainder = key[..., self.effective_head_dim:]
        else:
            query_to_rotate = query
            key_to_rotate = key

        # Reshape query/key to (B, num_heads, T, num_subvectors, 3)
        query_3d = query_to_rotate.view(batch_size, num_heads, seq_len, self.num_subvectors, 3)
        key_3d = key_to_rotate.view(batch_size, num_heads, seq_len, self.num_subvectors, 3)

        # Expand quaternion for broadcasting: (B, 1, T, num_subvectors)
        w = w.unsqueeze(1)
        qx = qx.unsqueeze(1)
        qy = qy.unsqueeze(1)
        qz = qz.unsqueeze(1)

        # Apply rotation to each 3D sub-vector
        query_rotated = self.quaternion_rotate(query_3d, w, qx, qy, qz)
        key_rotated = self.quaternion_rotate(key_3d, w, qx, qy, qz)

        # Reshape back to (B, num_heads, T, effective_head_dim)
        query_rotated = query_rotated.view(batch_size, num_heads, seq_len, self.effective_head_dim)
        key_rotated = key_rotated.view(batch_size, num_heads, seq_len, self.effective_head_dim)

        # Concatenate with non-rotated remainder if needed
        if self.needs_padding:
            query_rotated = torch.cat([query_rotated, query_remainder], dim=-1)
            key_rotated = torch.cat([key_rotated, key_remainder], dim=-1)

        return query_rotated, key_rotated


class LieAlgebraRelativeAttention(nn.Module):
    """
    Lie Algebra Relative Attention (LARA) for geometry-aware attention.

    Extends standard attention with geometric position awareness by combining:
    1. Semantic similarity (standard QK dot product)
    2. Geometric rotation (GeoPE quaternion rotation)
    3. Geometric bias (learned MLP from relative displacement)

    The attention score decomposes into three components (from GeoPE paper):
    - Projected Similarity: <q, k> cos(A) - standard similarity modulated by distance
    - Axial Alignment: (q·n)(k·n)(1-cos(A)) - sensitivity to displacement direction
    - Torsional Component: (n×q)·k sin(A) - captures relative orientation

    For EDA routing, this enables attention based on both:
    - Token semantics (direction type, special tokens)
    - Spatial proximity in chip coordinates

    Args:
        hidden_size: Input/output hidden dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
        dropout: Attention dropout probability
        use_geometric_bias: Whether to add MLP-based geometric bias
        bias_mlp_hidden: Hidden dimension for bias MLP

    Example:
        >>> lara = LieAlgebraRelativeAttention(hidden_size=256, num_heads=4)
        >>> hidden = torch.randn(2, 100, 256)
        >>> coords = torch.randn(2, 100, 3) * 10000
        >>> output = lara(hidden, coords)
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 4,
        head_dim: int = 64,
        dropout: float = 0.1,
        use_geometric_bias: bool = True,
        bias_mlp_hidden: int = 64,
        coord_scale: float = 1e-4,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_geometric_bias = use_geometric_bias
        self.coord_scale = coord_scale

        # Standard attention projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size)

        # Geometric position embedding (quaternion rotation)
        self.geo_pe = GeometricPositionEmbedding(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            coord_scale=coord_scale,
        )

        # Geometric bias MLP (optional)
        # Maps relative 3D displacement to per-head attention bias
        if use_geometric_bias:
            self.geo_bias_mlp = nn.Sequential(
                nn.Linear(3, bias_mlp_hidden),
                nn.GELU(),
                nn.Linear(bias_mlp_hidden, num_heads),
            )

        self.dropout = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5

    def compute_geometric_bias(
        self,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention bias based on 3D relative displacements.

        This adds direction-aware geometric priors to attention:
        - Nearby tokens in chip space get higher attention
        - Direction of displacement matters (not just distance)
        - Each head can learn different spatial preferences

        Args:
            coordinates: 3D coordinates (batch, seq_len, 3)

        Returns:
            Attention bias (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = coordinates.shape

        # Compute pairwise relative displacement
        # Δp_ij = p_i - p_j for all pairs
        coords_i = coordinates.unsqueeze(2)  # (B, T, 1, 3)
        coords_j = coordinates.unsqueeze(1)  # (B, 1, T, 3)
        rel_disp = coords_i - coords_j       # (B, T, T, 3)

        # Scale relative displacement
        rel_disp = rel_disp.float() * self.coord_scale

        # MLP to compute per-head bias
        bias = self.geo_bias_mlp(rel_disp)  # (B, T, T, num_heads)
        bias = bias.permute(0, 3, 1, 2)      # (B, num_heads, T, T)

        return bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        coordinates: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Geometry-aware attention forward pass.

        Args:
            hidden_states: Input hidden states (batch, seq_len, hidden_size)
            coordinates: 3D coordinates (batch, seq_len, 3)
            attention_mask: Optional attention mask (batch, seq_len) or (batch, seq_len, seq_len)
            output_attentions: Whether to return attention weights

        Returns:
            Tuple of (output,) or (output, attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        # Transpose for attention: (B, num_heads, T, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Apply geometric rotation (GeoPE)
        query_rot, key_rot = self.geo_pe(query, key, coordinates)

        # Compute attention scores with rotated Q, K
        attn_scores = torch.matmul(query_rot, key_rot.transpose(-2, -1)) * self.scale
        # (B, num_heads, T, T)

        # Add geometric bias
        if self.use_geometric_bias:
            geo_bias = self.compute_geometric_bias(coordinates)
            attn_scores = attn_scores + geo_bias

        # Apply attention mask
        if attention_mask is not None:
            # Expand mask: (B, 1, 1, T) for 2D mask or (B, 1, T, T) for 3D mask
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(
                attention_mask == 0, float('-inf')
            )

        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        context = torch.matmul(attn_probs, value)  # (B, num_heads, T, head_dim)

        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(context)

        if output_attentions:
            return output, attn_probs
        return (output,)

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"use_geometric_bias={self.use_geometric_bias}"
        )
