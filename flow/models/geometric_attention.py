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
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F



def rotate_half(x):
    """Rotates half the hidden dims of the input (for RoPE)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand key/value heads for Grouped Query Attention (GQA).

    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_kv_heads, seq_len, head_dim)
    to (batch, num_attention_heads, seq_len, head_dim).
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding (RoPE) to query and key tensors.

    Args:
        q: Query tensor (batch, num_heads, seq_len, head_dim)
        k: Key tensor (batch, num_heads, seq_len, head_dim)
        cos: Cosine part of RoPE (batch, seq_len, head_dim)
        sin: Sine part of RoPE (batch, seq_len, head_dim)
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting

    Returns:
        Tuple of (q_rotated, k_rotated)
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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
        base: float = 32.0,
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

    def _prepare_coords(
        self,
        coordinates: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Prepare coordinates: handle batch/seq_len mismatches, scale to FP32.

        Args:
            coordinates: 3D coordinates (coord_batch, coord_seq, 3)
            batch_size: Expected batch size (from query/key)
            seq_len: Expected sequence length (from query/key)

        Returns:
            Scaled coordinates in FP32 (batch_size, seq_len, 3)
        """
        coords = coordinates.float()
        coord_batch, coord_seq, _ = coords.shape

        # Handle batch dimension mismatch (beam search expands batch by num_beams)
        if coord_batch != batch_size:
            if batch_size % coord_batch == 0:
                num_beams = batch_size // coord_batch
                coords = coords.unsqueeze(1).expand(-1, num_beams, -1, -1)
                coords = coords.reshape(batch_size, coord_seq, 3)
            else:
                raise ValueError(
                    f"Batch size mismatch: expected {batch_size}, coordinates has {coord_batch}"
                )

        # Handle seq_len mismatch (incremental decoding uses only current token)
        if coord_seq != seq_len:
            if coord_seq > seq_len:
                coords = coords[:, -seq_len:, :]
            else:
                raise ValueError(
                    f"Seq len mismatch: expected {seq_len}, coordinates has {coord_seq}"
                )

        # Scale coordinates in FP32
        coords = coords * self.coord_scale
        return coords

    def _coords_to_quaternion(
        self,
        coordinates: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert 3D coordinates to quaternion components for rotation.

        All computations in FP32 for numerical stability.

        Args:
            coordinates: 3D coordinates (batch, seq_len, 3)
            batch_size: Expected batch size
            seq_len: Expected sequence length

        Returns:
            Tuple of (w, qx, qy, qz), each (batch, 1, seq_len, num_subvectors)
            with head dimension already unsqueezed for broadcasting.
        """
        coords = self._prepare_coords(coordinates, batch_size, seq_len)

        x_pos = coords[:, :, 0]  # (B, T)
        y_pos = coords[:, :, 1]
        z_pos = coords[:, :, 2]

        # Compute phases for each sub-vector in FP32
        inv_freq = self.inv_freq  # (num_subvectors,) in FP32
        theta_x = torch.einsum('bt,f->btf', x_pos, inv_freq)  # (B, T, num_subvectors)
        theta_y = torch.einsum('bt,f->btf', y_pos, inv_freq)
        theta_z = torch.einsum('bt,f->btf', z_pos, inv_freq)

        # Compute symmetric quaternion (in FP32)
        w, qx, qy, qz = self.compute_quaternion(theta_x, theta_y, theta_z)

        # Expand for head broadcasting: (B, 1, T, num_subvectors)
        return w.unsqueeze(1), qx.unsqueeze(1), qy.unsqueeze(1), qz.unsqueeze(1)

    def _rotate_vectors(
        self,
        vectors: torch.Tensor,
        w: torch.Tensor,
        qx: torch.Tensor,
        qy: torch.Tensor,
        qz: torch.Tensor,
        inverse: bool = False,
    ) -> torch.Tensor:
        """
        Apply quaternion rotation to vectors, handling head_dim padding.

        Args:
            vectors: (batch, num_heads, seq_len, head_dim)
            w, qx, qy, qz: Quaternion components (batch, 1, seq_len, num_subvectors)
            inverse: If True, apply inverse rotation (conjugate quaternion: negate qx,qy,qz)

        Returns:
            Rotated vectors (batch, num_heads, seq_len, head_dim) in FP32
        """
        batch_size, num_heads, seq_len, head_dim = vectors.shape

        if inverse:
            qx, qy, qz = -qx, -qy, -qz

        if self.needs_padding:
            to_rotate = vectors[..., :self.effective_head_dim]
            remainder = vectors[..., self.effective_head_dim:]
        else:
            to_rotate = vectors

        vecs_3d = to_rotate.float().view(
            batch_size, num_heads, seq_len, self.num_subvectors, 3
        )
        rotated = self.quaternion_rotate(vecs_3d, w, qx, qy, qz)
        rotated = rotated.view(
            batch_size, num_heads, seq_len, self.effective_head_dim
        )

        if self.needs_padding:
            rotated = torch.cat([rotated, remainder.float()], dim=-1)

        return rotated

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_coordinates: torch.Tensor,
        key_coordinates: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply GeoPE rotation to query and key vectors.

        Supports both self-attention (single coordinate set) and cross-attention
        (separate coordinates for Q and K).

        In self-attention (key_coordinates=None):
            Q and K are rotated by the same coordinates. The relative geometry
            emerges from R(p_i)^{-1} R(p_j) in the dot product.

        In cross-attention (key_coordinates provided):
            Q is rotated by query_coordinates, K by key_coordinates. The relative
            geometry emerges from R(p^q_i)^{-1} R(p^kv_j).

        Args:
            query: Query vectors (batch, num_heads, query_len, head_dim)
            key: Key vectors (batch, num_heads, key_len, head_dim)
            query_coordinates: Coordinates for Q rotation (batch, query_len, 3)
            key_coordinates: Coordinates for K rotation (batch, key_len, 3).
                If None, uses query_coordinates (self-attention mode).

        Returns:
            Tuple of (query_rotated, key_rotated) with same shapes as input
        """
        input_dtype = query.dtype
        batch_size_q, num_heads, query_len, _ = query.shape
        batch_size_k, _, key_len, _ = key.shape

        # Compute quaternion for query coordinates
        w_q, qx_q, qy_q, qz_q = self._coords_to_quaternion(
            query_coordinates, batch_size_q, query_len
        )

        if key_coordinates is None:
            # Self-attention: same quaternion for Q and K
            w_k, qx_k, qy_k, qz_k = w_q, qx_q, qy_q, qz_q
        else:
            # Cross-attention: separate quaternion for K
            w_k, qx_k, qy_k, qz_k = self._coords_to_quaternion(
                key_coordinates, batch_size_k, key_len
            )

        # Rotate Q with query quaternion, K with key quaternion
        query_rotated = self._rotate_vectors(query, w_q, qx_q, qy_q, qz_q)
        key_rotated = self._rotate_vectors(key, w_k, qx_k, qy_k, qz_k)

        return query_rotated.to(input_dtype), key_rotated.to(input_dtype)

    def rotate_value(
        self,
        value: torch.Tensor,
        coordinates: torch.Tensor,
        inverse: bool = True,
    ) -> torch.Tensor:
        """
        Apply geometric rotation to value vectors for Cross-LARA value rotation.

        In Cross-LARA, value rotation uses a factored form:
            o_i = R(p^d_i) * Σ_j α_ij * R(p^e_j)^{-1} * v_j

        This method handles one side of the factorization:
        - rotate_value(V, p^e, inverse=True)  → R(p^e)^{-1} V  (encoder side, precomputed)
        - rotate_value(C, p^d, inverse=False)  → R(p^d) C       (decoder side, post-aggregation)

        Args:
            value: Value vectors (batch, num_heads, seq_len, head_dim)
            coordinates: 3D coordinates (batch, seq_len, 3)
            inverse: If True, apply inverse rotation R^{-1} (conjugate quaternion)

        Returns:
            Rotated value vectors (batch, num_heads, seq_len, head_dim)
        """
        input_dtype = value.dtype
        batch_size, num_heads, seq_len, _ = value.shape

        w, qx, qy, qz = self._coords_to_quaternion(coordinates, batch_size, seq_len)
        rotated = self._rotate_vectors(value, w, qx, qy, qz, inverse=inverse)

        return rotated.to(input_dtype)


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
        num_kv_heads: int = 4,
        head_dim: int = 64,
        dropout: float = 0.0,
        use_geometric_bias: bool = True,
        bias_mlp_hidden: int = 64,
        coord_scale: float = 1e-4,
        scaling: float = None,
        attn_logit_softcapping: float = None,
        attention_bias: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_geometric_bias = use_geometric_bias
        self.coord_scale = coord_scale
        self.attn_logit_softcapping = attn_logit_softcapping

        # GQA: number of query heads per key/value head group
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
        self.num_key_value_groups = num_heads // num_kv_heads

        # Attention scaling: matches T5Gemma's query_pre_attn_scalar**-0.5
        # Default: head_dim**-0.5 for backward compatibility
        self.scaling = scaling if scaling is not None else head_dim ** -0.5

        # Q projection: full num_heads (matches T5GemmaSelfAttention)
        # K/V projections: num_kv_heads for GQA (matches T5GemmaSelfAttention)
        # bias=attention_bias to match config.attention_bias (typically False)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=attention_bias)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=attention_bias)

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

        # attention_dropout: store as float, matching T5Gemma's convention.
        # Applied via F.dropout(x, p=self.attention_dropout if self.training else 0.0, training=True)
        # so eval mode ALWAYS gets p=0.0 regardless of the stored value.
        # Default is 0.0 (T5GemmaModuleConfig.attention_dropout = 0.0).
        self.attention_dropout = dropout

    def compute_geometric_bias(
        self,
        coordinates: torch.Tensor,
        target_dtype: torch.dtype = None,
    ) -> torch.Tensor:
        """
        Compute attention bias based on 3D relative displacements.

        This adds direction-aware geometric priors to attention:
        - Nearby tokens in chip space get higher attention
        - Direction of displacement matters (not just distance)
        - Each head can learn different spatial preferences

        Args:
            coordinates: 3D coordinates (batch, seq_len, 3)
            target_dtype: Target dtype for output (for fp16 compatibility)

        Returns:
            Attention bias (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = coordinates.shape
        if target_dtype is None:
            target_dtype = coordinates.dtype

        # Get MLP weight dtype for fp16 compatibility
        mlp_weight_dtype = self.geo_bias_mlp[0].weight.dtype

        # Compute pairwise relative displacement in FP32 first
        # Coordinates can be >65504 (FP16 max), so we must scale before converting
        coords_f32 = coordinates.float()
        coords_i = coords_f32.unsqueeze(2)  # (B, T, 1, 3)
        coords_j = coords_f32.unsqueeze(1)  # (B, 1, T, 3)
        rel_disp = (coords_i - coords_j) * self.coord_scale  # (B, T, T, 3)


        # Convert to MLP weight dtype after scaling
        rel_disp = rel_disp.to(mlp_weight_dtype)

        # MLP to compute per-head bias
        # Note: geo_bias_mlp weights may be fp16 in mixed precision training
        bias = self.geo_bias_mlp(rel_disp)  # (B, T, T, num_heads)
        bias = bias.permute(0, 3, 1, 2)      # (B, num_heads, T, T)

        # Cast to target dtype if needed
        if target_dtype is not None and bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
        return bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        coordinates: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Geometry-aware attention forward pass with RoPE + GeoPE.

        The rotation order is: RoPE (sequence position) → GeoPE (3D geometry)
        This allows the model to encode both sequential and spatial relationships.

        Args:
            hidden_states: Input hidden states (batch, seq_len, hidden_size)
            coordinates: 3D coordinates (batch, seq_len, 3)
            attention_mask: Optional attention mask (batch, seq_len) or (batch, seq_len, seq_len)
            output_attentions: Whether to return attention weights
            position_embeddings: RoPE embeddings tuple (cos, sin), each (batch, seq_len, head_dim)

        Returns:
            Tuple of (output,) or (output, attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Check dtype match before Linear calls
        if hidden_states.dtype != self.q_proj.weight.dtype:
            hidden_states = hidden_states.to(self.q_proj.weight.dtype)

        # Project to Q, K, V
        # Q: full num_heads, K/V: num_kv_heads (GQA, matches T5GemmaSelfAttention)
        query = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )

        # Transpose for attention: (B, num_heads, T, head_dim)
        query = query.transpose(1, 2)    # (B, num_heads, T, head_dim)
        key = key.transpose(1, 2)        # (B, num_kv_heads, T, head_dim)
        value = value.transpose(1, 2)    # (B, num_kv_heads, T, head_dim)

        # Step 1: Apply RoPE (Rotary Position Embedding) for sequence position
        # RoPE broadcasts over num_heads dimension, so works with different Q/K head counts
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Step 2: GQA expansion — expand K/V from num_kv_heads to num_heads via repetition
        # Must be done BEFORE GeoPE since GeoPE expects Q and K to have the same num_heads
        key = repeat_kv(key, self.num_key_value_groups)     # (B, num_heads, T, head_dim)
        value = repeat_kv(value, self.num_key_value_groups) # (B, num_heads, T, head_dim)

        # Step 3: Apply GeoPE (Geometric Position Embedding) for 3D spatial position
        # Both Q and K now have num_heads, so GeoPE works correctly
        # Self-attention: key_coordinates=None → same coords for Q and K
        query_rot, key_rot = self.geo_pe(query, key, query_coordinates=coordinates)
        
        # Step 4: Compute attention scores with rotated Q, K
        # Use self.scaling = query_pre_attn_scalar**-0.5 (matches T5Gemma, NOT head_dim**-0.5)
        attn_scores = torch.matmul(query_rot, key_rot.transpose(-2, -1)) * self.scaling
        # (B, num_heads, T, T)

        # Step 5: Add geometric bias (if enabled)
        if self.use_geometric_bias:
            geo_bias = self.compute_geometric_bias(coordinates, target_dtype=attn_scores.dtype)
            attn_scores = attn_scores + geo_bias

        # Step 6: Attention logit soft-capping (matches T5Gemma's attn_logit_softcapping=50.0)
        # Caps attention logits to [-cap, cap] via tanh to prevent degenerate softmax.
        # Critical for training stability with random initialization.
        if self.attn_logit_softcapping is not None:
            attn_scores = attn_scores / self.attn_logit_softcapping
            attn_scores = torch.tanh(attn_scores)
            attn_scores = attn_scores * self.attn_logit_softcapping

        # Step 7: Apply attention mask (additive format: 0=valid, large_negative=masked)
        if attention_mask is not None:
            # Slice to match key length (matches T5Gemma's eager_attention_forward)
            causal_mask = attention_mask[:, :, :, :key_rot.shape[-2]]
            attn_scores = attn_scores + causal_mask

        # Step 8: Softmax in FP32 for numerical stability (matches T5Gemma's eager_attention_forward)
        # FP16 softmax can overflow/underflow with large attention scores
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_rot.dtype)
        # Attention dropout — matches T5Gemma: p=0.0 in eval, p=attention_dropout in train
        # Using explicit self.training check (same pattern as T5GemmaSelfAttention.forward)
        attn_probs = F.dropout(
            attn_probs,
            p=self.attention_dropout if self.training else 0.0,
            training=True,
        )

        # Step 9: Apply attention to values
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


class T5GemmaLARAAttention(nn.Module):
    """
    LARA Attention wrapper compatible with T5GemmaSelfAttention interface.

    This wrapper adapts LieAlgebraRelativeAttention to work as a drop-in
    replacement for T5GemmaSelfAttention in T5Gemma encoder/decoder layers.

    Key adaptations:
    - Accepts position_embeddings (RoPE cos/sin) and applies both RoPE and GeoPE
    - Accepts coordinates via kwargs for geometric attention
    - Returns format: (attn_output, attn_weights, present_key_value)
    - Supports output_attentions flag

    Position Encoding Strategy (RoPE + GeoPE):
    - RoPE: Encodes sequential token positions (order in sequence)
    - GeoPE: Encodes 3D spatial positions (chip coordinates)
    - Both are applied as rotations: Q, K → RoPE rotation → GeoPE rotation → attention

    Args:
        config: T5GemmaConfig or T5GemmaModuleConfig
        layer_idx: Layer index in the model
        coord_scale: Scaling factor for input coordinates
        use_geometric_bias: Whether to use MLP-based geometric bias
        bias_mlp_hidden: Hidden dimension for geometric bias MLP

    Note:
        - KV caching (past_key_value) is not yet supported
        - Flash Attention is not yet supported
        - Coordinates parameter is required in forward pass
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

        # Extract config parameters — support both T5GemmaModuleConfig and T5GemmaConfig
        if hasattr(config, 'hidden_size'):
            # T5GemmaModuleConfig (decoder sub-config) — the typical case for decoder layers
            self.hidden_size = config.hidden_size
            self.num_heads = config.num_attention_heads
            self.num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
            self.head_dim = config.head_dim
            # Matches T5GemmaSelfAttention: scaling = query_pre_attn_scalar**-0.5
            query_pre_attn_scalar = getattr(config, 'query_pre_attn_scalar', self.head_dim)
            # attention_dropout: specifically for attention weights (typically 0.0 in T5Gemma)
            # Distinct from residual dropout_rate (0.1), which is NOT applied to attention weights
            attention_dropout = getattr(config, 'attention_dropout', 0.0)
            # Soft-cap attention logits to [-cap, cap] via tanh (T5Gemma default: 50.0)
            attn_logit_softcapping = getattr(config, 'attn_logit_softcapping', None)
            # Attention projection bias (T5Gemma default: False)
            attention_bias = getattr(config, 'attention_bias', False)
        elif hasattr(config, 'encoder'):
            # Top-level T5GemmaConfig — fallback path
            encoder_config = config.encoder
            self.hidden_size = encoder_config.hidden_size
            self.num_heads = encoder_config.num_attention_heads
            self.num_kv_heads = getattr(encoder_config, 'num_key_value_heads', self.num_heads)
            self.head_dim = encoder_config.head_dim
            query_pre_attn_scalar = getattr(encoder_config, 'query_pre_attn_scalar', self.head_dim)
            attention_dropout = getattr(encoder_config, 'attention_dropout', 0.0)
            attn_logit_softcapping = getattr(encoder_config, 'attn_logit_softcapping', None)
            attention_bias = getattr(encoder_config, 'attention_bias', False)
        else:
            raise ValueError(f"Unexpected config type: {type(config)}")

        self.layer_idx = layer_idx
        self.coord_scale = coord_scale

        # Core LARA module — initialized with T5Gemma-consistent parameters
        self.lara = LieAlgebraRelativeAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dropout=attention_dropout,
            use_geometric_bias=use_geometric_bias,
            bias_mlp_hidden=bias_mlp_hidden,
            coord_scale=coord_scale,
            scaling=query_pre_attn_scalar ** -0.5,
            attn_logit_softcapping=attn_logit_softcapping,
            attention_bias=attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        coordinates: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass compatible with T5GemmaSelfAttention interface.

        Args:
            hidden_states: Input hidden states (batch, seq_len, hidden_size)
            position_embeddings: RoPE embeddings (cos, sin) - Applied before GeoPE
            attention_mask: Attention mask (batch, seq_len) or (batch, seq_len, seq_len)
            past_key_value: KV cache - Not yet supported
            cache_position: Cache position indices - Not yet supported
            output_attentions: Whether to return attention weights
            coordinates: 3D coordinates (batch, seq_len, 3) - REQUIRED for LARA

        Returns:
            Tuple of (attn_output, attn_weights, present_key_value)
            - attn_output: (batch, seq_len, hidden_size)
            - attn_weights: (batch, num_heads, seq_len, seq_len) if output_attentions else None
            - present_key_value: None (caching not supported)

        Raises:
            ValueError: If coordinates is None
            NotImplementedError: If past_key_value is provided
        """

        # Validate required parameters
        if coordinates is None:
            raise ValueError(
                "T5GemmaLARAAttention requires 'coordinates' parameter. "
                "Make sure to pass encoder_abs_positions through the model forward pass. "
                f"Layer index: {self.layer_idx}"
            )

        # Check for unsupported features
        if past_key_value is not None:
            raise NotImplementedError(
                f"KV caching not yet supported in T5GemmaLARAAttention (layer {self.layer_idx}). "
                "Set use_cache=False in model.generate() or model.forward()."
            )

        # Process attention mask
        # T5Gemma passes 4D additive mask (batch, 1, seq_len, seq_len)
        # LARA now expects 4D additive mask for direct addition to attn_scores
        # Keep 4D format, no squeeze needed

        # Call LARA forward with both RoPE and GeoPE
        # Rotation order: RoPE (sequence position) → GeoPE (3D geometry)
        lara_output = self.lara(
            hidden_states=hidden_states,
            coordinates=coordinates,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,  # Pass RoPE to LARA
        )

        # Parse output
        if output_attentions:
            attn_output, attn_weights = lara_output
        else:
            attn_output = lara_output[0]
            attn_weights = None

        # Return format compatible with T5GemmaSelfAttention
        # (attn_output, attn_weights, present_key_value)
        present_key_value = None  # No caching support
        return attn_output, attn_weights, present_key_value

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"layer_idx={self.layer_idx}, "
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"coord_scale={self.coord_scale}"
        )


class CrossLARAAttention(nn.Module):
    """
    Cross-Attention with Lie Algebra Relative Attention (Cross-LARA).

    Extends LARA to cross-attention with three levels of geometric awareness:

    Level 1 — WHERE to attend (geometric bias):
        Learned MLP bias from relative displacement φ(p^d_i - p^e_j) modulates
        attention scores, providing a spatial prior for which pins to attend to.

    Level 2 — HOW to interact (dual-coordinate GeoPE on Q/K):
        Q is rotated by decoder coordinates, K by encoder coordinates:
            q̂_i = R(p^d_i) q_i,  k̂_j = R(p^e_j) k_j
        The dot product ⟨q̂_i, k̂_j⟩ = ⟨q_i, R_rel k_j⟩ decomposes via Rodrigues
        into projected similarity + axial alignment + torsional components,
        capturing direction-sensitive geometric relationships.

    Level 3 — WHAT to extract (geometric value rotation):
        Values are aggregated through a factored reference-frame transform:
            o_i = R(p^d_i) · Σ_j α_ij · R(p^e_j)^{-1} · v_j
        Encoder values are "unrotated" to a canonical frame, aggregated, then
        rotated into the decoder's local reference frame. This allows the model
        to extract different semantic information depending on relative geometry.

    Key properties:
    - Geometric Coincidence: When p^d_i = p^e_j (decoder at a pin), all rotations
      cancel → pure semantic attention + constant bias. The model focuses on
      "what information does this pin carry", not "where is it".
    - Self-attention Reduction: When P^q = P^{kv}, value rotation cancels to
      identity, recovering exactly the Self-LARA formulation.
    - Efficiency: Value rotation uses a factored form with O((T_d + T_e) · d)
      extra cost, same asymptotic complexity as standard attention.

    For EDA routing:
    - At pin positions: pure semantic attention (read pin properties)
    - Between pins: geometrically modulated attention (nearby pins contribute
      fine-grained routing guidance, far pins contribute connectivity constraints)

    Args:
        config: Model config with hidden_size, num_heads, etc.
        layer_idx: Layer index
        coord_scale: Coordinate scaling factor
        use_geometric_bias: Whether to use MLP-based geometric bias (Level 1)
        use_value_rotation: Whether to use geometric value rotation (Level 3)
        bias_mlp_hidden: Hidden dimension for bias MLP
        cross_attention_hidden_size: Encoder hidden size (if different from decoder)
    """

    def __init__(
        self,
        config,
        layer_idx: int = 0,
        coord_scale: float = 1e-5,
        use_geometric_bias: bool = True,
        use_value_rotation: bool = True,
        bias_mlp_hidden: int = 64,
        cross_attention_hidden_size: Optional[int] = None,
    ):
        super().__init__()

        # Extract config parameters — support both T5GemmaModuleConfig and T5GemmaConfig
        if hasattr(config, 'hidden_size'):
            # T5GemmaModuleConfig (decoder sub-config)
            self.hidden_size = config.hidden_size
            self.num_heads = config.num_attention_heads
            self.num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
            self.head_dim = config.head_dim
            query_pre_attn_scalar = getattr(config, 'query_pre_attn_scalar', self.head_dim)
            attention_dropout = getattr(config, 'attention_dropout', 0.0)
            attn_logit_softcapping = getattr(config, 'attn_logit_softcapping', None)
            attention_bias = getattr(config, 'attention_bias', False)
        elif hasattr(config, 'decoder'):
            # Top-level T5GemmaConfig
            decoder_config = config.decoder
            self.hidden_size = decoder_config.hidden_size
            self.num_heads = decoder_config.num_attention_heads
            self.num_kv_heads = getattr(decoder_config, 'num_key_value_heads', self.num_heads)
            self.head_dim = decoder_config.head_dim
            query_pre_attn_scalar = getattr(decoder_config, 'query_pre_attn_scalar', self.head_dim)
            attention_dropout = getattr(decoder_config, 'attention_dropout', 0.0)
            attn_logit_softcapping = getattr(decoder_config, 'attn_logit_softcapping', None)
            attention_bias = getattr(decoder_config, 'attention_bias', False)
        else:
            raise ValueError(f"Unexpected config type: {type(config)}")

        self.layer_idx = layer_idx
        self.coord_scale = coord_scale
        self.use_geometric_bias = use_geometric_bias
        self.use_value_rotation = use_value_rotation
        self.attn_logit_softcapping = attn_logit_softcapping

        # GQA: num_heads may differ from num_kv_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        # Attention scaling: query_pre_attn_scalar**-0.5 (matches T5Gemma)
        self.scaling = query_pre_attn_scalar ** -0.5

        # Cross-attention may have different hidden sizes for encoder/decoder
        self.cross_attention_hidden_size = cross_attention_hidden_size or self.hidden_size

        # Q projection: full num_heads; K/V: num_kv_heads for GQA
        # bias=attention_bias to match config (typically False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(self.cross_attention_hidden_size, self.num_kv_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(self.cross_attention_hidden_size, self.num_kv_heads * self.head_dim, bias=attention_bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=attention_bias)

        # Geometric position embedding (for Q/K rotation and value rotation)
        self.geo_pe = GeometricPositionEmbedding(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            coord_scale=coord_scale,
        )

        # Geometric bias MLP (Level 1)
        if use_geometric_bias:
            self.geo_bias_mlp = nn.Sequential(
                nn.Linear(3, bias_mlp_hidden),
                nn.GELU(),
                nn.Linear(bias_mlp_hidden, self.num_heads),
            )

        # Store as float, same convention as T5Gemma (0.0 by default)
        self.attention_dropout = attention_dropout

    def compute_cross_geometric_bias(
        self,
        query_coords: torch.Tensor,
        key_coords: torch.Tensor,
        target_dtype: torch.dtype = None,
    ) -> torch.Tensor:
        """
        Compute geometric bias for cross-attention (Level 1).

        Maps relative 3D displacement to per-head attention bias:
            b_{ij}^{(h)} = MLP(p^d_i - p^e_j)

        Args:
            query_coords: Decoder coordinates (batch, query_len, 3)
            key_coords: Encoder coordinates (batch, key_len, 3)
            target_dtype: Target dtype for output (for fp16 compatibility)

        Returns:
            Geometric bias (batch, num_heads, query_len, key_len)
        """
        if target_dtype is None:
            target_dtype = query_coords.dtype

        # Get MLP weight dtype for fp16 compatibility
        mlp_weight_dtype = self.geo_bias_mlp[0].weight.dtype

        # Compute pairwise relative displacement in FP32 first
        # Coordinates can be >65504 (FP16 max), so we must scale before converting
        q_coords_f32 = query_coords.float().unsqueeze(2)  # (batch, query_len, 1, 3)
        k_coords_f32 = key_coords.float().unsqueeze(1)    # (batch, 1, key_len, 3)
        rel_disp = (q_coords_f32 - k_coords_f32) * self.coord_scale  # (batch, query_len, key_len, 3)

        # Convert to MLP weight dtype after scaling
        rel_disp = rel_disp.to(mlp_weight_dtype)

        # MLP to compute per-head bias
        bias = self.geo_bias_mlp(rel_disp)  # (batch, query_len, key_len, num_heads)
        bias = bias.permute(0, 3, 1, 2)      # (batch, num_heads, query_len, key_len)

        if target_dtype is not None and bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
        return bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        query_coordinates: Optional[torch.Tensor] = None,
        key_coordinates: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Cross-LARA forward pass with three levels of geometric awareness.

        Implements the unified LARA formula for cross-attention:
            o_i = R(p^d_i) · softmax(R(p^d_i)Q · (R(p^e_j)K)^T / √d + φ(Δp)) · R(p^e_j)^{-1} V

        Args:
            hidden_states: Decoder hidden states (batch, query_len, hidden_size)
            encoder_hidden_states: Encoder hidden states (batch, key_len, hidden_size)
            attention_mask: Attention mask (additive: 0=valid, large_neg=masked)
            query_coordinates: Decoder coordinates (batch, query_len, 3)
            key_coordinates: Encoder coordinates (batch, key_len, 3)
            past_key_value: KV cache (not yet supported)
            output_attentions: Whether to return attention weights

        Returns:
            (output, attention_weights, present_key_value)
        """
        batch_size, query_len, _ = hidden_states.shape
        _, key_len, _ = encoder_hidden_states.shape

        # Check dtype match before Linear calls
        if hidden_states.dtype != self.q_proj.weight.dtype:
            hidden_states = hidden_states.to(self.q_proj.weight.dtype)
        if encoder_hidden_states.dtype != self.k_proj.weight.dtype:
            encoder_hidden_states = encoder_hidden_states.to(self.k_proj.weight.dtype)

        # Validate coordinates
        if query_coordinates is None or key_coordinates is None:
            raise ValueError(
                "CrossLARAAttention requires both query_coordinates and key_coordinates. "
                f"Got query_coordinates={query_coordinates is not None}, "
                f"key_coordinates={key_coordinates is not None}"
            )

        if past_key_value is not None:
            raise NotImplementedError("KV caching not yet supported in CrossLARAAttention")

        # ── Step 1: Linear projections ──
        # Q from decoder, K/V from encoder (GQA: K/V use num_kv_heads)
        query = self.q_proj(hidden_states).view(
            batch_size, query_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # (batch, num_heads, query_len, head_dim)

        key = self.k_proj(encoder_hidden_states).view(
            batch_size, key_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)  # (batch, num_kv_heads, key_len, head_dim)

        value = self.v_proj(encoder_hidden_states).view(
            batch_size, key_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)  # (batch, num_kv_heads, key_len, head_dim)

        # ── Step 2: GQA expansion ──
        # Expand K/V from num_kv_heads to num_heads before GeoPE
        key = repeat_kv(key, self.num_key_value_groups)     # (batch, num_heads, key_len, head_dim)
        value = repeat_kv(value, self.num_key_value_groups) # (batch, num_heads, key_len, head_dim)

        # ── Step 3: Level 2 — Dual-coordinate GeoPE on Q/K ──
        # Q rotated by decoder coords, K rotated by encoder coords
        # Attention score decomposes via Rodrigues formula:
        #   ⟨R(p^d_i)q, R(p^e_j)k⟩ = ⟨q, R_rel k⟩
        # where R_rel = R(p^d_i)^{-1} R(p^e_j) encodes relative geometry
        query_rot, key_rot = self.geo_pe(
            query, key,
            query_coordinates=query_coordinates,
            key_coordinates=key_coordinates,
        )

        # ── Step 4: Attention scores ──
        attn_scores = torch.matmul(query_rot, key_rot.transpose(-2, -1)) * self.scaling
        # (batch, num_heads, query_len, key_len)

        # ── Step 5: Level 1 — Geometric bias ──
        if self.use_geometric_bias:
            geo_bias = self.compute_cross_geometric_bias(
                query_coordinates, key_coordinates, target_dtype=attn_scores.dtype
            )
            attn_scores = attn_scores + geo_bias

        # ── Step 6: Attention logit soft-capping ──
        if self.attn_logit_softcapping is not None:
            attn_scores = attn_scores / self.attn_logit_softcapping
            attn_scores = torch.tanh(attn_scores)
            attn_scores = attn_scores * self.attn_logit_softcapping

        # ── Step 7: Attention mask ──
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # ── Step 8: Softmax + dropout ──
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_rot.dtype)
        attn_probs = F.dropout(
            attn_probs,
            p=self.attention_dropout if self.training else 0.0,
            training=True,
        )

        # ── Step 9: Level 3 — Geometric value rotation ──
        # Factored form: o_i = R(p^d_i) · Σ_j α_ij · R(p^e_j)^{-1} · v_j
        #
        # Encoder side (precomputed):  ṽ_j = R(p^e_j)^{-1} v_j     — O(T_e · d)
        # Standard aggregation:       c_i = Σ_j α_ij · ṽ_j          — O(T_d · T_e · d)
        # Decoder side (post-agg):    o_i = R(p^d_i) · c_i           — O(T_d · d)
        #
        # Geometric Coincidence: when p^d_i = p^e_j, R(p^d)·R(p^e)^{-1} = I
        # → value passes through unchanged (pure semantic attention at pin positions)
        if self.use_value_rotation:
            # Encoder side: unrotate values to canonical frame
            value_canonical = self.geo_pe.rotate_value(
                value, key_coordinates, inverse=True
            )  # R(p^e_j)^{-1} v_j

            # Standard weighted aggregation in canonical frame
            context = torch.matmul(attn_probs, value_canonical)
            # (batch, num_heads, query_len, head_dim)

            # Decoder side: rotate aggregated context to decoder's reference frame
            context = self.geo_pe.rotate_value(
                context, query_coordinates, inverse=False
            )  # R(p^d_i) · c_i
        else:
            # Standard value aggregation (no geometric value rotation)
            context = torch.matmul(attn_probs, value)

        # ── Step 10: Output projection ──
        context = context.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.num_heads * self.head_dim
        )
        output = self.out_proj(context)

        present_key_value = None
        attn_weights = attn_probs if output_attentions else None

        return output, attn_weights, present_key_value
