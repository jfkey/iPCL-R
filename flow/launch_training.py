#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   launch_training.py
@Time    :   2025/08/01 11:16:30
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Command-line launcher for model training using Accelerate
             framework, handles distributed training setup and hyperparameter
             configuration with FlowConfig integration
"""

import argparse
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch

from accelerate import Accelerator, InitProcessGroupKwargs


def _pin_process_to_local_gpu() -> None:
    """
    Force each distributed rank's default CUDA device to its own local GPU
    *before* anything else allocates on cuda:0.

    During `trainer.train(resume_from_checkpoint=...)`, HF Trainer calls
    `torch.load(...)` on optimizer / scheduler / RNG state files. Those
    tensors were serialized with their original device (often cuda:0), so
    without an explicit map_location every rank reconstructs them on the
    physical cuda:0 -> 16 * ~310MiB piled on GPU 0 -> OOM on resume.

    Setting the current device here makes torch.load default to the current
    device for tensors that were saved on a CUDA device, avoiding the pileup.
    """
    if not torch.cuda.is_available():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    if local_rank < torch.cuda.device_count():
        torch.cuda.set_device(local_rank)


def _patch_torch_load_for_resume() -> None:
    """
    Wrap torch.load so any call without an explicit map_location defaults to
    CPU. This prevents every rank's checkpoint deserialization from landing
    on physical cuda:0 (tensors were serialized with their original device,
    usually cuda:0) during resume_from_checkpoint. Optimizer state tensors
    are later moved to the correct GPU automatically by
    optimizer.load_state_dict based on the parameter's device; RNG states
    are expected to stay on CPU as ByteTensor.
    """
    _orig_load = torch.load

    def _load(*args, **kwargs):
        if "map_location" not in kwargs:
            kwargs["map_location"] = "cpu"
        return _orig_load(*args, **kwargs)

    torch.load = _load

from flow.config import FlowConfig
from flow.training import TrainingPipeline
from flow.utils import setup_logging


def main():
    setup_logging()
    """Main entry point for training launcher"""
    parser = argparse.ArgumentParser(description="Launch training using FlowConfig")
    parser.add_argument(
        "--flow-config", type=str, required=True, help="Path to FlowConfig JSON file"
    )

    args = parser.parse_args()

    try:
        # Pin this rank to its GPU and make torch.load default to that GPU so
        # resume_from_checkpoint doesn't pile every rank's checkpoint tensors
        # onto physical cuda:0 (which caused OOM on resume).
        _pin_process_to_local_gpu()
        _patch_torch_load_for_resume()

        # Increase NCCL timeout to 60 min for first-run dataset preprocessing
        # (dataset.map inside main_process_first can exceed the 10 min default)
        accelerator = Accelerator(
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=60))]
        )
        # Load FlowConfig from JSON file
        logging.info(f"🔧 Loading configuration from: {args.flow_config}")
        flow_config = FlowConfig.from_config_file(Path(args.flow_config))

        # Extract training stage configuration directly
        tokenization_config = flow_config.tokenization
        training_config = flow_config.training

        logging.info("🚀 Starting training pipeline")
        logging.info(f"   Token dataset: {tokenization_config.paths.token_dataset_dir}")
        logging.info(f"   Model save dir: {training_config.paths.model_save_dir}")
        logging.info(
            f"   Training epochs: {training_config.hyperparameters.num_train_epochs}"
        )
        logging.info(
            f"   Batch size per device: {training_config.hyperparameters.batch_size_per_device}"
        )
        logging.info(
            f"   Learning rate: {training_config.hyperparameters.learning_rate}"
        )
        logging.info(f"   Optimizer: {training_config.hyperparameters.optimizer_type}")

        # Create and run training pipeline
        pipeline = TrainingPipeline(accelerator, flow_config)

        start_time = time.time()
        pipeline.run_training()
        end_time = time.time()

        # Report results
        execution_time = end_time - start_time
        logging.info("✅ Training completed successfully!")
        logging.info(f"   Execution time: {execution_time:.2f} seconds")
        logging.info(f"   Model saved to: {training_config.paths.model_save_dir}")
        logging.info(
            f"   Split dataset saved to: {training_config.paths.split_dataset_dir}"
        )

        logging.info(f"   Logs saved to: {training_config.paths.logging_dir}")
        logging.info("💡 Next step: Run evaluation with:")
        logging.info(
            f"   accelerate launch -m flow.launch_evaluation --flow-config {args.flow_config}"
        )

    except FileNotFoundError as e:
        logging.info(f"❌ Error: Configuration file not found: {e}")
        return 1
    except KeyError as e:
        logging.info(f"❌ Error: Missing configuration key: {e}")
        return 1
    except Exception as e:
        logging.info(f"❌ Training failed with error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

# Usage: 
# accelerate launch -m  --config_file /home/liujunfeng/.cache/huggingface/accelerate/default_config.yaml  flow.launch_training --flow-config /mnt/local_data1/liujunfeng/exp/Large-GeoPE/stage_training/model_wope/config.json 