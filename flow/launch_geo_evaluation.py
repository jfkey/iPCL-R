#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   launch_geo_evaluation.py
@Time    :   2025/01/30
@Author  :   Junfeng Liu
@Version :   3.0
@Desc    :   Evaluation script for GeoT5GemmaForConditionalGeneration.

             Follows the same pattern as launch_evaluation.py (T5Gemma):
             - Flow config.json controls paths, data, generation params
             - Model config (including geometric_config) is read from checkpoint

             Automatically branches behavior based on model.geo_config:
             - Baseline (all geo off): identical to launch_evaluation.py
             - PE only (no LARA):      standard generate with encoder coordinates
             - LARA enabled:           per-sample generate with coordinate tracking

             Supports unseen design evaluation via eval_designs in config.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

import torch
from accelerate import Accelerator, PartialState
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GenerationConfig

from flow.config import FlowConfig
from flow.evaluation import EvaluationPipeline
from flow.models.geo_t5gemma import GeoT5GemmaForConditionalGeneration
from flow.tokenization import TokenizationPipeline, UnifiedTokenizer
from flow.utils import load_corpus_dataset, setup_logging


# =============================================================================
#  Geo feature detection
# =============================================================================

def detect_geo_mode(model: GeoT5GemmaForConditionalGeneration):
    """Detect which geo features are active from the loaded model.

    Returns:
        (has_pe, has_lara) -- two booleans
    """
    geo = model.geo_config
    has_pe = geo.use_advanced_geo_pe
    has_lara = geo.use_geo_self_attn or geo.use_geo_cross_attn
    return has_pe, has_lara


# =============================================================================
#  Data collators
# =============================================================================

def make_baseline_collator(tokenizer, max_src_len):
    """Same collator as launch_evaluation.py -- tokens only."""
    def collate_fn(batch):
        source_tokens = [item["source_tokens"] for item in batch]
        encs = tokenizer(
            source_tokens,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            padding="max_length",
            max_length=max_src_len,
        )
        return encs
    return collate_fn


class GeoEvalCollator:
    """Collator that pads and scales encoder relative coordinates to FP16.

    Scale values must match those used during training (from model's geo_config).
    """

    def __init__(self, tokenizer, max_src_len, coord_scale, coord_scale_z):
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.coord_scale = coord_scale
        self.coord_scale_z = coord_scale_z

    def __call__(self, batch):
        source_tokens = [item["source_tokens"] for item in batch]
        encs = self.tokenizer(
            source_tokens,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            padding="max_length",
            max_length=self.max_src_len,
        )

        batch_size = len(batch)
        padded_src_len = encs["input_ids"].shape[1]

        # Pad in FP32 for safe scaling
        encoder_rel_positions = torch.zeros(
            batch_size, padded_src_len, 3, dtype=torch.float32
        )

        for i, item in enumerate(batch):
            if "src_rel_pos" in item:
                src_rel = item["src_rel_pos"]
                seq_len = min(len(src_rel), padded_src_len)
                for j in range(seq_len):
                    if isinstance(src_rel[j], (list, tuple)) and len(src_rel[j]) >= 3:
                        encoder_rel_positions[i, j, 0] = src_rel[j][0]
                        encoder_rel_positions[i, j, 1] = src_rel[j][1]
                        encoder_rel_positions[i, j, 2] = src_rel[j][2]

        # Per-axis scaling (must match training collator) and convert to FP16
        encoder_rel_positions[:, :, 0] *= self.coord_scale
        encoder_rel_positions[:, :, 1] *= self.coord_scale
        encoder_rel_positions[:, :, 2] *= self.coord_scale_z
        encoder_rel_positions = encoder_rel_positions.half()

        return {
            "input_ids": encs["input_ids"],
            "attention_mask": encs["attention_mask"],
            "encoder_rel_positions": encoder_rel_positions,
        }


# =============================================================================
#  Load components
# =============================================================================

def load_components(config: FlowConfig):
    """Load dataset, tokenizer, and model.

    Model is loaded via from_pretrained -- geo_config comes from checkpoint.
    Dataset is optionally filtered by eval_designs from config.
    """
    # Dataset
    dataset_config = config.dataset
    target_split = dataset_config.validation_split
    dataset = load_corpus_dataset(dataset_config, split=target_split)

    # Filter by eval_designs if specified
    eval_designs = config.evaluation.eval_designs
    if eval_designs:
        dataset = dataset.filter(lambda x: x["source_design"] in eval_designs)
        logging.info(f"Filtered to designs {eval_designs}: {len(dataset)} samples")
    else:
        logging.info(f"Using all designs: {len(dataset)} samples")

    tokenization_pipeline = TokenizationPipeline(config)
    dataset = tokenization_pipeline.preprocess_corpus(dataset)
    dataset = tokenization_pipeline.build_token_dataset(dataset, remove_columns=False)

    dataset_source = (
        dataset_config.hub_id
        if dataset_config.use_hub()
        else dataset_config.local_path_for_split(target_split)
    )
    logging.info(f"Dataset loaded with {len(dataset)} samples from {dataset_source}")

    # Tokenizer
    unified_tokenizer = UnifiedTokenizer.from_pretrained(
        config.tokenization.paths.tokenizer_save_dir
    )
    tokenizer = unified_tokenizer.tokenizer
    logging.info(f"Tokenizer loaded from {config.tokenization.paths.tokenizer_save_dir}")

    # Model -- same pattern as launch_evaluation.py (T5Gemma)
    model_path = config.training.paths.model_save_dir
    try:
        model = GeoT5GemmaForConditionalGeneration.from_pretrained(model_path).eval()
        logging.info(f"Model loaded from {model_path}")
    except Exception as e:
        checkpoints = list(Path(model_path).glob("checkpoint-*"))
        if checkpoints:
            last_checkpoint = max(
                checkpoints, key=lambda x: int(x.name.split("-")[-1])
            )
            logging.info(f"Loading model from last checkpoint: {last_checkpoint}")
            model = GeoT5GemmaForConditionalGeneration.from_pretrained(
                last_checkpoint
            ).eval()
        else:
            logging.error(f"Failed to load model from {model_path}: {e}")
            raise e

    # DecimalBPE merger (None for other tokenizer algorithms). Needed at
    # inference time so the LARA coordinate tracker can expand merged tokens
    # like "R200_D300" into cumulative deltas.
    bpe_merger = getattr(unified_tokenizer, "bpe_merger", None)
    if bpe_merger is not None:
        logging.info(
            f"DecimalBPE merger loaded with {len(bpe_merger.merges)} merges"
        )

    return dataset, tokenizer, model, bpe_merger


# =============================================================================
#  LARA generation helper
# =============================================================================

def _generate_with_lara(
    model, tokenizer, input_ids, attention_mask,
    encoder_rel_positions, generation_config,
    bpe_merger=None,
) -> List[str]:
    """Per-sample generation with LARA coordinate tracking.

    LARA needs decoder coordinates updated at each step. The model's
    prepare_inputs_for_generation handles coordinate tracking and scaling
    via InferenceCoordinateTracker + geo_config scale values.

    When ``bpe_merger`` is provided (DecimalBPE), merged tokens such as
    ``R200_D300`` are expanded by the tracker so the cumulative delta is
    applied at each step.
    """
    batch_size = input_ids.shape[0]
    predictions = []

    # Set tokenizer for coordinate tracking (used by prepare_inputs_for_generation)
    model._tokenizer = tokenizer
    # Attach BPE merger (None for non-BPE algorithms) so the tracker created
    # inside model.generate() can expand merged tokens correctly.
    model._bpe_merger = bpe_merger

    for i in range(batch_size):
        with torch.no_grad():
            try:
                outputs = model.generate(
                    input_ids=input_ids[i:i+1],
                    attention_mask=attention_mask[i:i+1],
                    encoder_rel_positions=encoder_rel_positions[i:i+1],
                    driver_pos=(0, 0, 0),  # Relative coords: driver is origin
                    use_coordinate_tracking=True,
                    generation_config=generation_config,
                )
            except Exception as e:
                logging.warning(
                    f"Coordinate tracking failed for sample {i}: {e}, "
                    f"falling back to generate without tracking"
                )
                outputs = model.generate(
                    input_ids=input_ids[i:i+1],
                    attention_mask=attention_mask[i:i+1],
                    encoder_rel_positions=encoder_rel_positions[i:i+1],
                    generation_config=generation_config,
                )

        predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    return predictions


# =============================================================================
#  Run evaluation
# =============================================================================

def run_evaluation(config: FlowConfig):
    """Run evaluation -- branching on model.geo_config."""
    evaluation_config = config.evaluation
    training_hyperparameters_config = config.training.hyperparameters
    evaluation_paths_config = evaluation_config.paths
    dataset_config = config.dataset
    performance_config = evaluation_config.performance
    generation_config = evaluation_config.generation

    logging.info("Initializing evaluation pipeline")
    dataset_source = (
        dataset_config.hub_id
        if dataset_config.use_hub()
        else dataset_config.local_path_for_split(dataset_config.validation_split)
    )
    logging.info(f"   Dataset: {dataset_source} (split: {dataset_config.validation_split})")
    logging.info(f"   Tokenizer: {config.tokenization.paths.tokenizer_save_dir}")
    logging.info(f"   Model: {config.training.paths.model_save_dir}")
    logging.info(f"   Output: {evaluation_paths_config.output_dir}")
    if evaluation_config.eval_designs:
        logging.info(f"   Eval designs: {evaluation_config.eval_designs}")
    else:
        logging.info("   Eval designs: all")

    accelerator = Accelerator()

    # Load components
    with PartialState().main_process_first():
        dataset, tokenizer, model, bpe_merger = load_components(config)

    # Detect geo mode from loaded model
    has_pe, has_lara = detect_geo_mode(model)
    geo_active = has_pe or has_lara

    if has_lara:
        mode_str = "LARA (per-sample coordinate tracking, use_cache=False)"
    elif has_pe:
        mode_str = "PE only (batch generate with encoder coordinates)"
    else:
        mode_str = "Baseline (standard T5Gemma generate)"
    logging.info(f"   Geo mode: {mode_str}")

    # Select collator based on geo mode
    max_src_len = training_hyperparameters_config.max_src_len
    if geo_active:
        # Use scale values from model's geo_config (must match training)
        geo_cfg = model.geo_config
        collate_fn = GeoEvalCollator(
            tokenizer=tokenizer,
            max_src_len=max_src_len,
            coord_scale=geo_cfg.coord_scale,
            coord_scale_z=geo_cfg.coord_scale_z,
        )
        logging.info(
            f"   Coord scale: xy={geo_cfg.coord_scale}, z={geo_cfg.coord_scale_z}"
        )
    else:
        collate_fn = make_baseline_collator(tokenizer, max_src_len)

    dataloader = DataLoader(
        dataset,
        batch_size=performance_config.batch_size,
        num_workers=performance_config.dataloader_num_workers,
        pin_memory=performance_config.dataloader_pin_memory,
        collate_fn=collate_fn,
        shuffle=False,
    )

    model, dataloader = accelerator.prepare(model, dataloader)

    # Generation config -- use_cache depends on whether LARA is active
    model_generation_config = GenerationConfig(
        max_new_tokens=generation_config.max_new_tokens,
        num_beams=generation_config.num_beams,
        do_sample=generation_config.do_sample,
        temperature=generation_config.temperature if generation_config.do_sample else None,
        top_p=generation_config.top_p if generation_config.do_sample else None,
        top_k=generation_config.top_k if generation_config.do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
        use_cache=not has_lara,  # LARA does not support KV cache
    )

    logging.info(f"   Max new tokens: {generation_config.max_new_tokens}")
    logging.info(f"   Num beams: {generation_config.num_beams}")
    logging.info(f"   Do sample: {generation_config.do_sample}")
    logging.info(f"   use_cache: {not has_lara}")
    logging.info(f"   Batch size: {performance_config.batch_size}")

    # Run inference
    logging.info("Running inference...")
    inference_start_time = time.time()

    predictions = []
    for batch in tqdm(
        dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process
    ):
        unwrapped_model = accelerator.unwrap_model(model)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        if has_lara:
            # LARA: per-sample generation with coordinate tracking
            preds = _generate_with_lara(
                model=unwrapped_model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_rel_positions=batch["encoder_rel_positions"],
                generation_config=model_generation_config,
                bpe_merger=bpe_merger,
            )
        elif has_pe:
            # PE only: standard batch generate, but pass coordinates
            with torch.no_grad():
                outputs = unwrapped_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    encoder_rel_positions=batch["encoder_rel_positions"],
                    generation_config=model_generation_config,
                )
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        else:
            # Baseline: identical to launch_evaluation.py
            with torch.no_grad():
                outputs = unwrapped_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=model_generation_config,
                )
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        gathered_preds = accelerator.gather_for_metrics(preds)
        if accelerator.is_main_process:
            predictions.extend(gathered_preds)

    inference_time = time.time() - inference_start_time
    logging.info(f"Inference completed in {inference_time:.2f}s")
    logging.info(f"   Throughput: {len(dataset) / inference_time:.2f} samples/sec")

    # Metrics & save
    logging.info("Calculating evaluation metrics...")
    if accelerator.is_main_process:
        if len(predictions) != len(dataset):
            raise ValueError(
                f"Prediction count ({len(predictions)}) != dataset size ({len(dataset)})"
            )

        dataset = dataset.add_column("predictions", predictions)

        evaluation_pipeline = EvaluationPipeline(config)
        dataset = evaluation_pipeline.calculate_metrics(dataset)
        evaluation_pipeline.save_def_inference_metadata(dataset)
        evaluation_pipeline.save_def_inference_metadata_txt(dataset)

        logging.info(f"Saving results to {evaluation_paths_config.output_dir}")
        dataset.save_to_disk(evaluation_paths_config.output_dir)


# =============================================================================
#  Main
# =============================================================================

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

    try:
        flow_config = FlowConfig.from_config_file(Path(args.flow_config))
    except Exception as e:
        logging.info(f"Error loading config from {args.flow_config}: {e}")
        sys.exit(1)

    logging.info(f"Starting evaluation with config: {args.flow_config}")

    try:
        run_evaluation(flow_config)
        logging.info(
            f"Evaluation completed. Results: {flow_config.evaluation.paths.output_dir}"
        )
    except Exception as e:
        logging.info(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

# Usage:
# accelerate launch --config_file /home/liujunfeng/.cache/huggingface/accelerate/fast_evaluation.yaml -m flow.launch_geo_evaluation --flow-config flow_config.json
