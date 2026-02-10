#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Test VQ with FP16 mixed precision training.

This specifically tests the dtype handling that was causing:
"Index put requires the source and destination dtypes match, got Half for the destination and Float for the source."
"""

import torch
from transformers import T5GemmaConfig, T5GemmaModuleConfig
from flow.models.geo_t5gemma import GeoT5GemmaForConditionalGeneration, GeoConfig


def test_vq_fp16():
    """Test VQ with FP16 mixed precision."""
    print("=" * 80)
    print("Testing VQ with FP16 Mixed Precision")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping FP16 test")
        return

    device = torch.device("cuda")

    # Create model
    vocab_size = 100
    hidden_size = 256

    encoder_config = T5GemmaModuleConfig(
        hidden_size=hidden_size,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=128,
        sliding_window=64,
        vocab_size=vocab_size,
    )

    config = T5GemmaConfig(
        encoder=encoder_config,
        decoder=encoder_config,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    )

    geo_config = GeoConfig(
        use_advanced_geo_pe=True,
        use_vq=True,
        vq_codebook_size=64,
        vq_commitment_cost=0.25,
    )

    print("Creating model with VQ...")
    model = GeoT5GemmaForConditionalGeneration(config, geo_config)
    model = model.to(device)
    model.train()

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # FP16 scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    print("Running training with FP16 autocast...")

    for step in range(5):
        # Create batch
        batch_size = 4
        input_ids = torch.randint(0, vocab_size, (batch_size, 20), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, 15), device=device)
        attention_mask = torch.ones(batch_size, 20, device=device)

        encoder_abs_positions = torch.randint(0, 100000, (batch_size, 20, 3), device=device).float()
        encoder_rel_positions = torch.randint(-50000, 50000, (batch_size, 20, 3), device=device).float()

        # Forward with autocast (FP16)
        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                encoder_abs_positions=encoder_abs_positions,
                encoder_rel_positions=encoder_rel_positions,
            )

            loss = outputs.loss

        # Backward with gradient scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print(f"  Step {step}: loss = {loss.item():.4f}")

    print("\n✅ FP16 training successful! Dtype handling is correct.")

    # Verify VQ module dtypes
    print("\n" + "=" * 80)
    print("Verifying VQ Module Dtypes")
    print("=" * 80)

    vq = model.encoder_pe_vq

    # Create test input in FP16
    test_input_fp16 = torch.randn(2, 10, hidden_size, dtype=torch.float16, device=device)
    test_mask = torch.ones(2, 10, device=device)

    with torch.no_grad():
        quantized, vq_loss, indices = vq(test_input_fp16, test_mask)

    print(f"Input dtype: {test_input_fp16.dtype}")
    print(f"Output dtype: {quantized.dtype}")
    print(f"VQ loss dtype: {vq_loss.dtype}")

    assert quantized.dtype == test_input_fp16.dtype, "Output dtype should match input dtype"
    print("✓ Dtype preservation verified")

    # Test with FP32 input
    test_input_fp32 = torch.randn(2, 10, hidden_size, dtype=torch.float32, device=device)

    with torch.no_grad():
        quantized, vq_loss, indices = vq(test_input_fp32, test_mask)

    assert quantized.dtype == test_input_fp32.dtype, "Output dtype should match input dtype"
    print("✓ FP32 input works correctly")

    print("\n" + "=" * 80)
    print("✅ All FP16 tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_vq_fp16()
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
