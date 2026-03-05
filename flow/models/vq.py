#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   vq.py
@Time    :   2025/02/10
@Author  :   Junfeng Liu
@Version :   1.0
@Desc    :   Vector Quantization for Position Embeddings.

             Implements VQ-VAE style vector quantization to create an information
             bottleneck of log2(K) bits per token, preventing data leakage when
             adding position embeddings to token embeddings.

             Key features:
             - EMA codebook updates (no gradient needed for codebook)
             - Straight-through estimator for gradient flow
             - Dead code revival (reinitialize unused codes)
             - Proper handling of padding masks

             Reference:
             - "Neural Discrete Representation Learning" (VQ-VAE, NeurIPS 2017)
             - "Generating Diverse High-Fidelity Images with VQ-VAE-2" (NeurIPS 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class VectorQuantizer(nn.Module):
    """
    Vector Quantization with EMA codebook updates.

    Maps continuous position embedding vectors to a discrete codebook,
    creating an information bottleneck of log2(K) bits per token.

    Mathematical formulation:
        Input: z ∈ R^D (continuous PE vector)
        Codebook: E = {e_1, ..., e_K} where e_k ∈ R^D
        Quantization: q(z) = e_k* where k* = argmin_k ||z - e_k||²
        Output: q(z) ∈ R^D (quantized PE vector)

        Information: log2(K) bits (K codebook entries)

    Training:
        - Codebook updated via EMA (no gradient):
          e_k ← decay × e_k + (1 - decay) × mean(z : argmin ||z - e_j||² = k)
        - Encoder updated via commitment loss:
          L_commit = β × ||z - sg[q(z)]||²
        - Gradient flows through straight-through estimator:
          ∂L/∂z = ∂L/∂q(z)

    Args:
        hidden_size: Dimension of position embedding vectors (D)
        codebook_size: Number of codebook entries (K)
        commitment_cost: Weight β for commitment loss
        ema_decay: Exponential moving average decay rate
        dead_code_threshold: Usage threshold for dead code revival

    Example:
        >>> vq = VectorQuantizer(hidden_size=256, codebook_size=256)
        >>> pe = torch.randn(2, 100, 256)  # (batch, seq_len, hidden_size)
        >>> mask = torch.ones(2, 100)     # (batch, seq_len)
        >>> quantized_pe, vq_loss, indices = vq(pe, mask)
        >>> # quantized_pe: same shape as pe, but quantized to K entries
        >>> # vq_loss: scalar commitment loss
        >>> # indices: (2, 100) indices into codebook
    """

    def __init__(
        self,
        hidden_size: int,
        codebook_size: int = 256,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        dead_code_threshold: int = 2,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold

        # Codebook: K entries of D dimensions
        # No gradient needed: codebook is updated via EMA, not backprop.
        # Setting requires_grad=False also prevents DDP unused-parameter errors.
        self.embedding = nn.Embedding(codebook_size, hidden_size)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.embedding.weight.requires_grad_(False)

        # EMA tracking buffers (no gradient)
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_embedding_sum', self.embedding.weight.data.clone())
        self.register_buffer('initialized', torch.tensor(False))

        # Track code usage for dead code revival
        self.register_buffer('usage_count', torch.zeros(codebook_size))

    @torch.no_grad()
    def _initialize_codebook(self, flat_inputs: torch.Tensor):
        """
        Initialize codebook from first batch of data.

        Uses random selection from input vectors. This ensures the codebook
        starts in a reasonable range for the actual data distribution.

        Args:
            flat_inputs: Flattened input vectors (N, D)
        """
        n = flat_inputs.shape[0]

        if n >= self.codebook_size:
            # Random selection from inputs
            indices = torch.randperm(n, device=flat_inputs.device)[:self.codebook_size]
            self.embedding.weight.data.copy_(flat_inputs[indices])
        else:
            # Repeat and add small noise
            repeats = (self.codebook_size + n - 1) // n
            expanded = flat_inputs.repeat(repeats, 1)[:self.codebook_size]
            noise = torch.randn_like(expanded) * 0.01
            self.embedding.weight.data.copy_(expanded + noise)

        # Initialize EMA buffers
        self.ema_cluster_size.fill_(1.0)
        self.ema_embedding_sum.copy_(self.embedding.weight.data)
        self.initialized.fill_(True)

    @torch.no_grad()
    def _revive_dead_codes(self, flat_inputs: torch.Tensor):
        """
        Reinitialize codebook entries with low usage.

        Dead codes are entries that haven't been used frequently. This can
        happen when the codebook learns a suboptimal partition of the space.
        We reinitialize them with random samples from the current batch.

        Args:
            flat_inputs: Flattened input vectors (N, D)
        """
        dead_mask = self.usage_count < self.dead_code_threshold
        n_dead = dead_mask.sum().item()

        if n_dead == 0 or flat_inputs.shape[0] == 0:
            return

        # Sample from current batch to reinitialize
        n = flat_inputs.shape[0]
        indices = torch.randint(0, n, (n_dead,), device=flat_inputs.device)
        noise = torch.randn(n_dead, self.hidden_size, device=flat_inputs.device) * 0.01

        # Cast to match embedding weight dtype to avoid index_put dtype mismatch
        new_codes = (flat_inputs[indices] + noise).to(self.embedding.weight.dtype)
        self.embedding.weight.data[dead_mask] = new_codes

        # Reset EMA for revived codes (cast to match buffer dtypes)
        self.ema_cluster_size[dead_mask] = torch.ones(n_dead, device=self.ema_cluster_size.device, dtype=self.ema_cluster_size.dtype)
        self.ema_embedding_sum[dead_mask] = self.embedding.weight.data[dead_mask].to(self.ema_embedding_sum.dtype)
        self.usage_count[dead_mask] = torch.full((n_dead,), self.dead_code_threshold, device=self.usage_count.device, dtype=self.usage_count.dtype)

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with vector quantization.

        Args:
            inputs: Continuous PE vectors (batch, seq_len, hidden_size)
            attention_mask: Optional mask (batch, seq_len), 1=valid, 0=pad

        Returns:
            quantized: Quantized PE vectors (batch, seq_len, hidden_size)
            vq_loss: Scalar VQ loss (commitment loss)
            indices: Codebook indices (batch, seq_len)
        """
        batch_size, seq_len, hidden_size = inputs.shape
        input_dtype = inputs.dtype

        # Disable autocast to prevent unexpected FP16 conversions in VQ operations.
        # VQ requires precise FP32 distance computation, EMA updates, and index_put
        # operations that need consistent dtypes. Under autocast, ops like F.embedding
        # and matmul get cast to FP16, causing dtype mismatches in indexed assignments.
        device_type = 'cuda' if inputs.is_cuda else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            # Work in float32 for numerical stability in distance computation
            inputs_f32 = inputs.float()

            # Flatten: (B, T, D) -> (B*T, D)
            flat_inputs = inputs_f32.view(-1, hidden_size)

            # Extract valid positions (non-padding)
            if attention_mask is not None:
                valid_mask = attention_mask.bool().view(-1)  # (B*T,)
                flat_valid = flat_inputs[valid_mask]  # (N_valid, D)
            else:
                flat_valid = flat_inputs
                valid_mask = None

            # Lazy initialization from first batch
            if self.training and not self.initialized and flat_valid.shape[0] > 0:
                self._initialize_codebook(flat_valid.detach())

            # Get codebook in float32
            codebook = self.embedding.weight.float()  # (K, D)

            # Compute L2 distances: ||z - e||^2 = ||z||^2 - 2<z,e> + ||e||^2
            distances = (
                flat_valid.pow(2).sum(dim=-1, keepdim=True)  # ||z||^2
                - 2 * flat_valid @ codebook.t()               # -2<z,e>
                + codebook.pow(2).sum(dim=-1, keepdim=True).t()  # ||e||^2
            )  # (N_valid, K)

            # Find nearest codebook entry
            indices_valid = distances.argmin(dim=-1)  # (N_valid,)

            # Look up quantized vectors
            quantized_valid = F.embedding(indices_valid, codebook)  # (N_valid, D)

            # Commitment loss: pushes encoder output towards codebook
            # (codebook is updated via EMA, not gradient)
            commitment_loss = F.mse_loss(flat_valid, quantized_valid.detach())
            vq_loss = self.commitment_cost * commitment_loss

            # EMA codebook update (only during training)
            if self.training and flat_valid.shape[0] > 0:
                with torch.no_grad():
                    # One-hot encoding of assignments
                    encodings = F.one_hot(indices_valid, self.codebook_size).float()  # (N_valid, K)

                    # Update cluster sizes
                    cluster_size = encodings.sum(dim=0)  # (K,)
                    self.ema_cluster_size.mul_(self.ema_decay).add_(
                        cluster_size, alpha=1 - self.ema_decay
                    )

                    # Update embedding sums
                    embedding_sum = encodings.t() @ flat_valid  # (K, D)
                    self.ema_embedding_sum.mul_(self.ema_decay).add_(
                        embedding_sum, alpha=1 - self.ema_decay
                    )

                    # Laplace smoothing to prevent division by zero
                    n = self.ema_cluster_size.sum()
                    cluster_size_smooth = (
                        (self.ema_cluster_size + 1e-5)
                        / (n + self.codebook_size * 1e-5)
                        * n
                    )

                    # Update codebook
                    self.embedding.weight.data.copy_(
                        self.ema_embedding_sum / cluster_size_smooth.unsqueeze(-1)
                    )

                    # Track usage for dead code revival
                    self.usage_count.mul_(self.ema_decay).add_(
                        (cluster_size > 0).float(), alpha=1 - self.ema_decay
                    )

                    # Revive dead codes periodically
                    self._revive_dead_codes(flat_valid)

            # Straight-through estimator: forward uses quantized, backward uses input
            quantized_valid = flat_valid + (quantized_valid - flat_valid).detach()

            # Scatter back to full tensor (including padding positions)
            if valid_mask is not None:
                quantized_flat = torch.zeros(
                    batch_size * seq_len, hidden_size,
                    dtype=torch.float32,
                    device=inputs.device
                )
                quantized_flat[valid_mask] = quantized_valid

                indices_flat = torch.zeros(
                    batch_size * seq_len, dtype=torch.long, device=inputs.device
                )
                indices_flat[valid_mask] = indices_valid
            else:
                quantized_flat = quantized_valid
                indices_flat = indices_valid

            # Reshape back to original shape
            quantized = quantized_flat.view(batch_size, seq_len, hidden_size)
            indices = indices_flat.view(batch_size, seq_len)

        # Convert output to input dtype (e.g. fp16) outside autocast block
        quantized = quantized.to(input_dtype)

        return quantized, vq_loss, indices

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"codebook_size={self.codebook_size}, "
            f"commitment_cost={self.commitment_cost}, "
            f"ema_decay={self.ema_decay}"
        )
