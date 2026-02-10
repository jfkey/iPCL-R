#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for Metal Layer Only Position Embedding (GeometryAwarePositionEmbeddingTMP).

This script demonstrates how to enable and test the metal layer only PE for ablation study.
"""

import torch
from flow.models.geo_t5gemma import GeoConfig, GeoT5GemmaForConditionalGeneration
from transformers import T5GemmaConfig


def test_metal_layer_only_pe():
    """Test Metal Layer Only Position Embedding."""

    # Create a simple T5Gemma config
    config = T5GemmaConfig(
        vocab_size=32000,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
    )

    # Create GeoConfig with metal layer only PE enabled
    geo_config = GeoConfig(
        use_basic_fourier_pe=False,
        use_advanced_geo_pe=False,  # Disable full geometry-aware PE
        use_metal_layer_only_pe=True,  # Enable metal layer only PE
        num_frequencies=32,
        max_metal_layers=16,
        coord_scale=1e-5,
        pe_dropout=0.1,
    )

    print("=" * 80)
    print("Testing Metal Layer Only Position Embedding")
    print("=" * 80)
    print(f"\nGeoConfig settings:")
    print(f"  use_advanced_geo_pe: {geo_config.use_advanced_geo_pe}")
    print(f"  use_metal_layer_only_pe: {geo_config.use_metal_layer_only_pe}")
    print(f"  max_metal_layers: {geo_config.max_metal_layers}")
    print(f"  num_frequencies: {geo_config.num_frequencies}")

    # Create model with metal layer only PE
    model = GeoT5GemmaForConditionalGeneration(
        config=config,
        geo_config=geo_config,
    )

    print(f"\nModel encoder PE type: {type(model.encoder_geo_pe).__name__}")
    print(f"Model decoder PE type: {type(model.decoder_fourier_pe).__name__}")

    # Create sample input
    batch_size = 2
    seq_len = 10

    # Absolute positions: (x, y, metal_layer)
    abs_pos = torch.tensor([
        # Sequence 1: Different metal layers
        [
            [1000, 2000, 1],   # Layer 1 (vertical)
            [1500, 2500, 2],   # Layer 2 (horizontal)
            [2000, 3000, 3],   # Layer 3 (vertical)
            [2500, 3500, 4],   # Layer 4 (horizontal)
            [3000, 4000, 5],   # Layer 5 (vertical)
            [3500, 4500, 6],   # Layer 6 (horizontal)
            [4000, 5000, 7],   # Layer 7 (vertical)
            [4500, 5500, 8],   # Layer 8 (horizontal)
            [5000, 6000, 9],   # Layer 9 (vertical)
            [5500, 6500, 10],  # Layer 10 (horizontal)
        ],
        # Sequence 2: Same metal layers
        [
            [1000, 2000, 3],   # Layer 3 (vertical)
            [1500, 2500, 3],   # Layer 3 (vertical)
            [2000, 3000, 3],   # Layer 3 (vertical)
            [2500, 3500, 4],   # Layer 4 (horizontal)
            [3000, 4000, 4],   # Layer 4 (horizontal)
            [3500, 4500, 4],   # Layer 4 (horizontal)
            [4000, 5000, 5],   # Layer 5 (vertical)
            [4500, 5500, 5],   # Layer 5 (vertical)
            [5000, 6000, 5],   # Layer 5 (vertical)
            [5500, 6500, 5],   # Layer 5 (vertical)
        ],
    ], dtype=torch.float32)

    # Relative positions (not used in metal layer only PE, but required for interface)
    rel_pos = torch.zeros(batch_size, seq_len, 3)

    print(f"\nInput shapes:")
    print(f"  abs_pos: {abs_pos.shape}")
    print(f"  rel_pos: {rel_pos.shape}")

    # Forward pass through encoder PE
    with torch.no_grad():
        pe_output = model.encoder_geo_pe(
            abs_pos=abs_pos,
            rel_pos=rel_pos,
        )

    print(f"\nPosition Embedding output shape: {pe_output.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {config.hidden_size})")

    # Check that embeddings are different for different layers
    pe_seq1 = pe_output[0]  # (seq_len, hidden_size)

    # Compare embeddings for different layer types
    layer1_emb = pe_seq1[0]  # Layer 1 (vertical)
    layer2_emb = pe_seq1[1]  # Layer 2 (horizontal)
    layer3_emb = pe_seq1[2]  # Layer 3 (vertical)

    diff_12 = torch.norm(layer1_emb - layer2_emb).item()
    diff_13 = torch.norm(layer1_emb - layer3_emb).item()
    diff_23 = torch.norm(layer2_emb - layer3_emb).item()

    print(f"\nEmbedding differences (L2 norm):")
    print(f"  Layer 1 (vertical) vs Layer 2 (horizontal): {diff_12:.4f}")
    print(f"  Layer 1 (vertical) vs Layer 3 (vertical):   {diff_13:.4f}")
    print(f"  Layer 2 (horizontal) vs Layer 3 (vertical): {diff_23:.4f}")

    print(f"\nObservations:")
    print(f"  - Layers with same direction (1 vs 3) should have smaller difference")
    print(f"  - Layers with different direction (1 vs 2, 2 vs 3) should have larger difference")

    # Verify model has correct PE module
    assert model.encoder_geo_pe is not None, "Encoder should have geo PE"
    assert hasattr(model.encoder_geo_pe, 'layer_embed'), "Should have layer_embed"
    assert not hasattr(model.encoder_geo_pe, 'xy_fourier'), "Should NOT have xy_fourier"
    assert not hasattr(model.encoder_geo_pe, 'polar_rel'), "Should NOT have polar_rel"

    print(f"\n{'=' * 80}")
    print("✓ Metal Layer Only PE test passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_metal_layer_only_pe()
