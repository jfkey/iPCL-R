#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   launch_geo_evaluation.py
@Time    :   2025/01/30
@Author  :   Junfeng Liu
@Version :   1.0
@Desc    :   Command-line launcher for GeoT5GemmaForConditionalGeneration evaluation.

             This evaluation script handles the 5-field data format:
             - source_tokens: Encoder input tokens
             - target_tokens: Decoder target tokens (ground truth)
             - src_abs_pos: Absolute positions for <DRIVER>/<LOAD> tokens
             - src_rel_pos: Relative positions (load - driver) for <LOAD> tokens
             - tgt_coords: Generated in real-time using InferenceCoordinateTracker

             Key difference from launch_evaluation.py:
             - Uses GeoT5GemmaForConditionalGeneration instead of T5GemmaForConditionalGeneration
             - Handles coordinate inputs (encoder_abs_positions, encoder_rel_positions)
             - Uses InferenceCoordinateTracker for real-time tgt_coords generation during inference
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple, Union

import torch
from accelerate import Accelerator, PartialState
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    GenerationConfig,
    PreTrainedTokenizerFast,
)

from flow.config import FlowConfig
from flow.evaluation import EvaluationPipeline
from flow.tokenization import TokenizationPipeline, UnifiedTokenizer
from flow.utils import load_corpus_dataset, setup_logging
from flow.models.geo_t5gemma import GeoT5GemmaForConditionalGeneration


def extract_driver_position(src_abs_pos: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """
    Extract driver position from source absolute positions.

    The driver position is typically at index 1 (after <BOS> at index 0).
    We look for the first non-zero position in src_abs_pos.

    Args:
        src_abs_pos: List of (x, y, m) absolute positions for source tokens

    Returns:
        Driver position (x, y, m)
    """
    for pos in src_abs_pos:
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            x, y, m = pos[0], pos[1], pos[2]
            if x != 0 or y != 0 or m != 0:
                return (x, y, m)
    # Fallback: return origin if no driver position found
    return (0, 0, 0)


def load_components(
    config: FlowConfig,
) -> Union[Dataset, PreTrainedTokenizerFast, GeoT5GemmaForConditionalGeneration]:
    """Load evaluation dataset, tokenizer, and GeoT5Gemma model from config"""

    # Dataset
    dataset_config = config.dataset
    target_split = dataset_config.validation_split
    dataset = load_corpus_dataset(dataset_config, split=target_split)

    # dataset = dataset.select(range(min(100, len(dataset))))
    # logging.info(f"DEBUG: Using only {len(dataset)} samples for evaluation")

    tokenization_pipeline = TokenizationPipeline(config)
    dataset = tokenization_pipeline.preprocess_corpus(dataset)
    # Set remove_columns=False to keep original columns for evaluation ('source_design')
    dataset = tokenization_pipeline.build_token_dataset(dataset, remove_columns=False)

    dataset_source = (
        dataset_config.hub_id
        if dataset_config.use_hub()
        else dataset_config.local_path_for_split(target_split)
    )
    logging.info(f"Dataset loaded with {len(dataset)} samples from {dataset_source}")

    # Verify dataset has required coordinate columns
    required_columns = ["source_tokens", "target_tokens", "src_abs_pos", "src_rel_pos", "tgt_coords"]
    missing_columns = [col for col in required_columns if col not in dataset.column_names]
    if missing_columns:
        logging.warning(f"Dataset missing coordinate columns: {missing_columns}")
        logging.warning("Coordinate-aware evaluation may not work correctly.")
    else:
        logging.info("Dataset has all required coordinate columns for GeoT5Gemma evaluation")

    # Tokenizer & Model
    unified_tokenizer = UnifiedTokenizer.from_pretrained(
        config.tokenization.paths.tokenizer_save_dir
    )
    tokenizer = unified_tokenizer.tokenizer
    logging.info(
        f"Tokenizer loaded from {config.tokenization.paths.tokenizer_save_dir}"
    )

    # Try to load GeoT5GemmaForConditionalGeneration
    model_path = config.training.paths.model_save_dir
    try:
        model = GeoT5GemmaForConditionalGeneration.from_pretrained(model_path).eval()
        logging.info(f"GeoT5GemmaForConditionalGeneration loaded from {model_path}")
    except Exception as e:
        # Try loading from checkpoint
        checkpoints = list(Path(model_path).glob("checkpoint-*"))
        if checkpoints:
            last_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[-1]))
            logging.info(f"Loading GeoT5Gemma model from last checkpoint: {last_checkpoint}")
            model = GeoT5GemmaForConditionalGeneration.from_pretrained(last_checkpoint).eval()
        else:
            logging.error(f"Failed to load GeoT5Gemma model from {model_path}: {e}")
            raise e

    # Set tokenizer reference for coordinate tracking during generation
    model._tokenizer = tokenizer

    return dataset, tokenizer, model


class GeoDataCollatorForEvaluation:
    """
    Custom data collator for GeoT5Gemma evaluation.

    Handles:
    - source_tokens: Tokenized and padded to input_ids
    - src_abs_pos: Padded to encoder_abs_positions (batch, src_len, 3)
    - src_rel_pos: Padded to encoder_rel_positions (batch, src_len, 3)

    Note: tgt_coords is NOT included in the batch because it's generated
    in real-time during inference using InferenceCoordinateTracker.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_src_len: int,
        coord_pad_value: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.coord_pad_value = coord_pad_value

    def __call__(self, batch: List[dict]) -> dict:
        """Collate batch with coordinate handling."""
        source_tokens = [item["source_tokens"] for item in batch]

        # Tokenize source tokens
        encs = self.tokenizer(
            source_tokens,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            padding="max_length",
            max_length=self.max_src_len,
        )

        padded_src_len = encs["input_ids"].shape[1]
        batch_size = len(batch)

        # Pad source absolute positions
        encoder_abs_positions = torch.full(
            (batch_size, padded_src_len, 3),
            self.coord_pad_value,
            dtype=torch.long
        )

        # Pad source relative positions
        encoder_rel_positions = torch.full(
            (batch_size, padded_src_len, 3),
            self.coord_pad_value,
            dtype=torch.long
        )

        # Extract driver positions for coordinate tracking
        driver_positions = []

        for i, item in enumerate(batch):
            # Extract and pad src_abs_pos
            if "src_abs_pos" in item:
                src_abs_pos = item["src_abs_pos"]
                seq_len = min(len(src_abs_pos), padded_src_len)
                for j in range(seq_len):
                    if isinstance(src_abs_pos[j], (list, tuple)) and len(src_abs_pos[j]) >= 3:
                        encoder_abs_positions[i, j, 0] = src_abs_pos[j][0]
                        encoder_abs_positions[i, j, 1] = src_abs_pos[j][1]
                        encoder_abs_positions[i, j, 2] = src_abs_pos[j][2]

                # Extract driver position for this sample
                driver_pos = extract_driver_position(src_abs_pos)
                driver_positions.append(driver_pos)
            else:
                driver_positions.append((0, 0, 0)) 

            # Extract and pad src_rel_pos
            if "src_rel_pos" in item:
                src_rel_pos = item["src_rel_pos"]
                seq_len = min(len(src_rel_pos), padded_src_len)
                for j in range(seq_len):
                    if isinstance(src_rel_pos[j], (list, tuple)) and len(src_rel_pos[j]) >= 3:
                        encoder_rel_positions[i, j, 0] = src_rel_pos[j][0]
                        encoder_rel_positions[i, j, 1] = src_rel_pos[j][1]
                        encoder_rel_positions[i, j, 2] = src_rel_pos[j][2]

        return {
            "input_ids": encs["input_ids"],
            "attention_mask": encs["attention_mask"],
            "encoder_abs_positions": encoder_abs_positions,
            "encoder_rel_positions": encoder_rel_positions,
            "driver_positions": driver_positions,  # For coordinate tracking
        }


def check_model_supports_coordinates(model) -> bool:
    """Check if the model supports coordinate-aware generation."""
    from flow.models.geo_t5gemma import GeoT5GemmaDecoder

    if not hasattr(model, 'get_decoder'):
        return False

    decoder = model.get_decoder()
    return isinstance(decoder, GeoT5GemmaDecoder)


def generate_with_coordinates(
    model: GeoT5GemmaForConditionalGeneration,
    tokenizer: PreTrainedTokenizerFast,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    encoder_abs_positions: torch.Tensor,
    encoder_rel_positions: torch.Tensor,
    driver_positions: List[Tuple[int, int, int]],
    generation_config: GenerationConfig,
    use_coordinate_tracking: bool = True,
) -> List[str]:
    """
    Generate sequences with real-time coordinate tracking.

    This function handles batch generation where each sample may have
    a different driver position for coordinate tracking.

    For batch_size > 1, we process samples one at a time to properly
    track coordinates. For batch_size == 1, we use the model's built-in
    coordinate tracking.

    Args:
        model: GeoT5GemmaForConditionalGeneration model
        tokenizer: Tokenizer for decoding
        input_ids: Encoder input IDs (batch, src_len)
        attention_mask: Encoder attention mask (batch, src_len)
        encoder_abs_positions: Absolute positions (batch, src_len, 3)
        encoder_rel_positions: Relative positions (batch, src_len, 3)
        driver_positions: List of driver positions for each sample
        generation_config: Generation parameters
        use_coordinate_tracking: Whether to use real-time coordinate tracking

    Returns:
        List of decoded prediction strings
    """
    batch_size = input_ids.shape[0]
    device = input_ids.device
    predictions = []

    # Check if model supports coordinate tracking
    supports_coords = check_model_supports_coordinates(model)

    if not supports_coords:
        # Fallback: Generate without coordinate tracking
        logging.debug("Model does not support coordinate tracking, using standard generation")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_abs_positions=encoder_abs_positions,
                encoder_rel_positions=encoder_rel_positions,
                generation_config=generation_config,
            )

        for i in range(batch_size):
            pred = tokenizer.decode(outputs[i], skip_special_tokens=True)
            predictions.append(pred)
        return predictions

    # Process each sample individually for proper coordinate tracking
    # This is necessary because each sample has a different driver position
    for i in range(batch_size):
        # Extract single sample
        sample_input_ids = input_ids[i:i+1]
        sample_attention_mask = attention_mask[i:i+1]
        sample_encoder_abs = encoder_abs_positions[i:i+1]
        sample_encoder_rel = encoder_rel_positions[i:i+1]
        driver_pos = driver_positions[i]

        # Generate with coordinate tracking
        with torch.no_grad():
            try:
                outputs = model.generate(
                    input_ids=sample_input_ids,
                    attention_mask=sample_attention_mask,
                    encoder_abs_positions=sample_encoder_abs,
                    encoder_rel_positions=sample_encoder_rel,
                    driver_pos=driver_pos,
                    use_coordinate_tracking=use_coordinate_tracking,
                    generation_config=generation_config,
                )
            except Exception as e:
                # Fallback if coordinate tracking fails
                logging.warning(f"Coordinate tracking failed for sample {i}: {e}, using fallback")
                outputs = model.generate(
                    input_ids=sample_input_ids,
                    attention_mask=sample_attention_mask,
                    encoder_abs_positions=sample_encoder_abs,
                    encoder_rel_positions=sample_encoder_rel,
                    generation_config=generation_config,
                )

        # Decode prediction
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(pred)

    return predictions


def run_evaluation(config: FlowConfig):
    """Run GeoT5Gemma evaluation with coordinate support"""
    evaluation_config = config.evaluation
    training_paths_config = config.training.paths
    training_hyperparameters_config = config.training.hyperparameters
    tokenization_paths_config = config.tokenization.paths
    evaluation_paths_config = evaluation_config.paths
    dataset_config = config.dataset
    performance_config = evaluation_config.performance
    generation_config = evaluation_config.generation

    logging.info("🚀 Initializing GeoT5Gemma evaluation pipeline")
    dataset_source = (
        dataset_config.hub_id
        if dataset_config.use_hub()
        else dataset_config.local_path_for_split(dataset_config.validation_split)
    )
    logging.info(
        f"   Dataset: {dataset_source} (split: {dataset_config.validation_split})"
    )
    logging.info(f"   Tokenizer: {tokenization_paths_config.tokenizer_save_dir}")
    logging.info(f"   Model: {training_paths_config.model_save_dir}")
    logging.info(f"   Output: {evaluation_paths_config.output_dir}")

    accelerator = Accelerator()

    # Load dataset, tokenizer, and model
    with PartialState().main_process_first():
        dataset, tokenizer, model = load_components(config)
        logging.info("✅ GeoT5Gemma components loaded successfully")

    # Create custom collator for GeoT5Gemma
    collate_fn = GeoDataCollatorForEvaluation(
        tokenizer=tokenizer,
        max_src_len=training_hyperparameters_config.max_src_len,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=performance_config.batch_size,
        num_workers=performance_config.dataloader_num_workers,
        pin_memory=performance_config.dataloader_pin_memory,
        collate_fn=collate_fn,
        shuffle=False,
    )

    model, dataloader = accelerator.prepare(model, dataloader)

    # Configure generation parameters
    # NOTE: use_cache=False is REQUIRED for LARA attention because:
    # - LARA recomputes K,V from hidden_states each pass (no KV caching support)
    # - With use_cache=True, HuggingFace passes only the last token's hidden_states
    #   but attention_mask covers all tokens, causing shape mismatch in matmul
    model_generation_config = GenerationConfig(
        max_new_tokens=generation_config.max_new_tokens,
        num_beams=generation_config.num_beams,
        do_sample=generation_config.do_sample,
        temperature=generation_config.temperature
        if generation_config.do_sample
        else None,
        top_p=generation_config.top_p if generation_config.do_sample else None,
        top_k=generation_config.top_k if generation_config.do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
        use_cache=False,  # Required for LARA - no KV caching support
    )

    logging.info("🎯 Generation parameters:")
    logging.info(f"   Max new tokens: {generation_config.max_new_tokens}")
    logging.info(f"   Beam search: {generation_config.num_beams}")
    logging.info(f"   Do sample: {generation_config.do_sample}")
    if generation_config.do_sample:
        logging.info(f"   Temperature: {generation_config.temperature}")
        logging.info(f"   Top-p: {generation_config.top_p}")
        logging.info(f"   Top-k: {generation_config.top_k}")
    logging.info(f"   Batch size: {performance_config.batch_size}")

    # Check coordinate tracking support
    supports_coord_tracking = check_model_supports_coordinates(accelerator.unwrap_model(model))
    if supports_coord_tracking:
        logging.info("   Coordinate tracking: Enabled (InferenceCoordinateTracker + LARA)")
        # Warn about LARA + beam search compatibility
        if generation_config.num_beams > 1:
            logging.warning("⚠️  LARA attention does not support KV caching.")
            logging.warning("   With beam search (num_beams > 1), LARA decoder may not work optimally.")
            logging.warning("   Consider setting num_beams=1 for LARA models, or disable LARA (use_geo_self_attn=False).")
    else:
        logging.info("   Coordinate tracking: Disabled (model uses standard attention)")
        logging.info("   Note: Encoder position embeddings are still used for geometry-aware input")

    # Print sample data for verification
    if accelerator.is_main_process:
        sample = dataset[0]
        logging.info("\n📋 Sample data format verification:")
        logging.info(f"   source_tokens: {sample['source_tokens'][:100]}...")
        if 'src_abs_pos' in sample:
            driver_pos = extract_driver_position(sample['src_abs_pos'])
            logging.info(f"   driver_pos extracted: {driver_pos}")
            logging.info(f"   src_abs_pos[0:5]: {sample['src_abs_pos'][:5]}")
        if 'src_rel_pos' in sample:
            logging.info(f"   src_rel_pos[0:5]: {sample['src_rel_pos'][:5]}")
        if 'tgt_coords' in sample:
            logging.info(f"   tgt_coords[0:5] (ground truth): {sample['tgt_coords'][:5]}")

    # Run inference
    logging.info("⚡ Running GeoT5Gemma inference with coordinate tracking...")
    inference_start_time = time.time()

    predictions = []
    for batch in tqdm(
        dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process
    ):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        encoder_abs_positions = batch["encoder_abs_positions"]
        encoder_rel_positions = batch["encoder_rel_positions"]
        driver_positions = batch["driver_positions"]

        # Generate with coordinate tracking
        preds = generate_with_coordinates(
            model=accelerator.unwrap_model(model),
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_abs_positions=encoder_abs_positions,
            encoder_rel_positions=encoder_rel_positions,
            driver_positions=driver_positions,
            generation_config=model_generation_config,
        )

        gathered_preds = accelerator.gather_for_metrics(preds)

        if accelerator.is_main_process:
            predictions.extend(gathered_preds)

    inference_time = time.time() - inference_start_time
    logging.info(f"✅ Inference completed in {inference_time:.2f}s")
    logging.info(f"   Throughput: {len(dataset) / inference_time:.2f} samples/sec")

    # Use existing metrics system for evaluation
    logging.info("📊 Calculating evaluation metrics...")

    if accelerator.is_main_process:
        # Sort predictions by original indices to ensure correct order
        if len(predictions) != len(dataset):
            logging.error(
                f"Number of predictions ({len(predictions)}) does not match dataset size ({len(dataset)}). Truncating."
            )
            raise ValueError("Inconsistent prediction and dataset sizes")

        dataset = dataset.add_column("predictions", predictions)

        # Evaluation pipeline
        evaluation_pipeline = EvaluationPipeline(config)
        dataset = evaluation_pipeline.calculate_metrics(dataset)

        # Save DEF inference metadata for EDA tool evaluation
        evaluation_pipeline.save_def_inference_metadata(dataset)
        evaluation_pipeline.save_def_inference_metadata_txt(dataset)

        # Save results
        logging.info("💾 Saving evaluation results...")
        dataset.save_to_disk(evaluation_paths_config.output_dir)


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Launch GeoT5Gemma evaluation")
    parser.add_argument(
        "--flow-config",
        type=str,
        required=True,
        help="Path to flow configuration JSON file",
    )

    args = parser.parse_args()

    # Load flow configuration
    try:
        flow_config = FlowConfig.from_config_file(Path(args.flow_config))
    except Exception as e:
        logging.info(f"❌ Error loading config from {args.flow_config}: {e}")
        sys.exit(1)

    logging.info("🔍 Starting GeoT5Gemma evaluation")
    logging.info(f"   Config: {args.flow_config}")

    try:
        # Run evaluation
        run_evaluation(flow_config)

        logging.info("✅ GeoT5Gemma evaluation completed successfully!")
        logging.info(f"   Results saved to: {flow_config.evaluation.paths.output_dir}")

    except Exception as e:
        logging.info(f"❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

# Usage:1
# accelerate launch --config_file /home/liujunfeng/.cache/huggingface/accelerate/fast_evaluation.yaml -m flow.launch_geo_evaluation --flow-config config.json