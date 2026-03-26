#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data_collator.py
@Time    :   2025/01/14
@Author  :   Dawn Li
@Version :   1.0
@Desc    :   Data collator for geometry-aware trajectory generation.
             Extends DataCollatorForSeq2Seq to handle 3D coordinates alongside
             token sequences for GeoT5Gemma training.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase


@dataclass
class GeoDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    """
    Data collator for sequence-to-sequence models with position embedding support.

    Extends DataCollatorForSeq2Seq to handle 3D coordinate tensors alongside
    token sequences. Coordinates are scaled and converted to FP16 in the collator
    so that downstream model code receives ready-to-use tensors without further
    scaling or FP32 conversion.

    Position fields consumed from dataset features:
    - src_rel_pos: Relative positions for source tokens (load - driver)
      - <DRIVER> token: (0, 0, 0)
      - <LOAD> tokens: (load - driver) relative position
      - Other tokens: (0, 0, 0)
    - relative_tgt_coords: Relative cumulative positions for target tokens
      (each coordinate relative to the driver position)

    Scaling applied here (coord_scale for x,y; coord_scale_z for z):
      scaled_coord = (x * coord_scale, y * coord_scale, z * coord_scale_z)
    After scaling, values are O(1) and safe for FP16 (max 65504).

    Output batch includes:
    - input_ids, attention_mask, labels (standard Seq2Seq)
    - encoder_rel_positions: Scaled FP16 relative positions (batch, src_len, 3)
    - decoder_coordinates: Scaled FP16 relative positions (batch, tgt_len, 3)

    Args:
        tokenizer: Tokenizer for padding
        model: Model for label preparation
        padding: Padding strategy
        max_length: Maximum sequence length
        pad_to_multiple_of: Pad to multiple of this value
        label_pad_token_id: Token ID to pad labels with (-100 for loss ignore)
        return_tensors: Output tensor type ('pt' for PyTorch)
        coord_pad_value: Value to pad coordinates with (default 0)
        coord_scale: Scaling factor for x,y axes (default 1e-3)
        coord_scale_z: Scaling factor for z axis / metal layer (default 1.0)
    """

    coord_pad_value: int = 0
    coord_scale: float = 1e-3
    coord_scale_z: float = 1.0

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        # Extract coordinate fields before parent collator (it doesn't know them)
        src_rel_pos_list = []
        tgt_coords_list = []
        has_src_rel_pos = False
        has_tgt_coords = False

        for feature in features:
            # Pop coordinate fields so parent collator ignores them
            if "src_rel_pos" in feature:
                src_rel_pos_list.append(feature.pop("src_rel_pos"))
                has_src_rel_pos = True
            if "relative_tgt_coords" in feature:
                tgt_coords_list.append(feature.pop("relative_tgt_coords"))
                has_tgt_coords = True
            # Remove src_abs_pos / tgt_coords if present (no longer used)
            feature.pop("src_abs_pos", None)
            feature.pop("tgt_coords", None)

        # Call parent collator for standard Seq2Seq processing
        batch = super().__call__(features, return_tensors=return_tensors)

        # Get padded sequence lengths
        padded_src_len = batch["input_ids"].shape[1]
        padded_tgt_len = batch["labels"].shape[1] if "labels" in batch else None

        # Pad, scale, and convert to FP16
        if has_src_rel_pos and src_rel_pos_list:
            batch["encoder_rel_positions"] = self._pad_and_scale_coordinates(
                src_rel_pos_list, padded_src_len
            )

        if has_tgt_coords and tgt_coords_list:
            if padded_tgt_len is None:
                padded_tgt_len = max(len(coords) for coords in tgt_coords_list)
            batch["decoder_coordinates"] = self._pad_and_scale_coordinates(
                tgt_coords_list, padded_tgt_len
            )

        return batch

    def _pad_and_scale_coordinates(
        self,
        coords_list: List[List[tuple]],
        target_length: int,
    ) -> torch.Tensor:
        """
        Pad coordinate sequences, apply per-axis scaling, and convert to FP16.

        Scaling is done in FP32 (raw chip coords can exceed FP16 range),
        then cast to FP16 since scaled values are O(1).

        Args:
            coords_list: List of coordinate sequences, each is List[(x, y, z)]
            target_length: Target sequence length after padding

        Returns:
            Scaled padded coordinate tensor (batch_size, target_length, 3) in FP16
        """
        batch_size = len(coords_list)
        # Pad in FP32 first for safe scaling
        padded = torch.zeros(batch_size, target_length, 3, dtype=torch.float32)

        for i, coords in enumerate(coords_list):
            seq_len = min(len(coords), target_length)
            for j in range(seq_len):
                if isinstance(coords[j], (list, tuple)) and len(coords[j]) >= 3:
                    padded[i, j, 0] = coords[j][0]
                    padded[i, j, 1] = coords[j][1]
                    padded[i, j, 2] = coords[j][2]

        # Per-axis scaling: x,y use coord_scale, z uses coord_scale_z
        padded[:, :, 0] *= self.coord_scale
        padded[:, :, 1] *= self.coord_scale
        padded[:, :, 2] *= self.coord_scale_z

        # Convert to FP16 — values are now O(1), safe for half precision
        return padded.half()


def create_geo_data_collator(
    tokenizer: PreTrainedTokenizerBase,
    model: Optional[Any] = None,
    pad_to_multiple_of: int = 8,
    label_pad_token_id: int = -100,
    coord_pad_value: int = 0,
    coord_scale: float = 1e-6,
    coord_scale_z: float = 0.3,
) -> GeoDataCollatorForSeq2Seq:
    """
    Create a GeoDataCollatorForSeq2Seq with standard settings.

    Args:
        tokenizer: Tokenizer for padding
        model: Model for label preparation
        pad_to_multiple_of: Pad sequences to multiple of this value
        label_pad_token_id: Token ID for padding labels (-100 to ignore in loss)
        coord_pad_value: Value for padding coordinates
        coord_scale: Scaling factor for x,y coordinate axes
        coord_scale_z: Scaling factor for z axis (metal layer)

    Returns:
        Configured GeoDataCollatorForSeq2Seq instance
    """
    return GeoDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=pad_to_multiple_of,
        padding=True,
        label_pad_token_id=label_pad_token_id,
        coord_pad_value=coord_pad_value,
        coord_scale=coord_scale,
        coord_scale_z=coord_scale_z,
    )


def add_coordinates_to_dataset(
    dataset,
    source_col: str = "source_tokens",
    target_col: str = "target_tokens",
    driver_col: str = "driver",
    num_workers: int = 16,
):
    """
    Add coordinate columns to a HuggingFace dataset.

    NOTE: This function is DEPRECATED. Coordinates are now computed during
    the tokenization pipeline stage. Use the tokenization pipeline instead.

    This function processes a dataset to compute and add source_coords
    and target_coords columns based on the token sequences.

    Args:
        dataset: HuggingFace Dataset with source_tokens and target_tokens
        source_col: Name of source tokens column
        target_col: Name of target tokens column
        driver_col: Name of driver coordinate column
        num_workers: Number of parallel workers for processing

    Returns:
        Dataset with added source_coords and target_coords columns

    Example:
        >>> dataset = load_from_disk("token_dataset")
        >>> dataset = add_coordinates_to_dataset(dataset)
        >>> # Now dataset has 'source_coords' and 'target_coords' columns
    """
    import warnings
    warnings.warn(
        "add_coordinates_to_dataset is deprecated. "
        "Coordinates are now computed during tokenization pipeline. "
        "Re-run tokenization to include coordinates in the dataset.",
        DeprecationWarning,
        stacklevel=2,
    )

    from flow.tokenization.coordinate_utils import (
        compute_coordinates_for_tokenized_sample,
    )

    def compute_coords(batch):
        """Compute coordinates for a batch of samples."""
        source_coords_batch = []
        target_coords_batch = []

        drivers = batch.get(driver_col, [None] * len(batch[source_col]))

        for source_tokens, target_tokens, driver in zip(
            batch[source_col], batch[target_col], drivers
        ):
            source_coords, target_coords = compute_coordinates_for_tokenized_sample(
                source_tokens, target_tokens, driver_str=driver
            )
            source_coords_batch.append(source_coords)
            target_coords_batch.append(target_coords)

        batch["source_coords"] = source_coords_batch
        batch["target_coords"] = target_coords_batch
        return batch

    # Apply coordinate computation
    dataset = dataset.map(
        compute_coords,
        batched=True,
        num_proc=num_workers,
        desc="Computing coordinates",
    )

    return dataset
