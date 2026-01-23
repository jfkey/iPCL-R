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

    Extends DataCollatorForSeq2Seq to handle 3D position tensors alongside
    token sequences. Positions are padded to match the padded sequence length.

    Position Embedding Design:
    - src_abs_pos: Absolute positions for source tokens
      - <DRIVER> token gets driver's absolute position
      - <LOAD> tokens get load's absolute position
      - Other tokens get (0, 0, 0)
    - src_rel_pos: Relative positions for source tokens
      - <DRIVER> token gets (0, 0, 0)
      - <LOAD> tokens get (load - driver) relative position
      - Other tokens get (0, 0, 0)
    - tgt_coords: Cumulative absolute positions for target tokens

    Expected input features:
    - input_ids: Encoder input token IDs
    - attention_mask: Encoder attention mask
    - labels: Decoder target token IDs
    - src_abs_pos: List of (x, y, m) absolute positions for encoder tokens
    - src_rel_pos: List of (dx, dy, dm) relative positions for encoder tokens
    - tgt_coords: List of (x, y, m) cumulative positions for decoder tokens

    Output batch includes:
    - input_ids, attention_mask, labels (standard Seq2Seq)
    - encoder_abs_positions: Padded absolute positions (batch, src_len, 3)
    - encoder_rel_positions: Padded relative positions (batch, src_len, 3)
    - decoder_coordinates: Padded cumulative positions (batch, tgt_len, 3)

    Args:
        tokenizer: Tokenizer for padding
        model: Model for label preparation
        padding: Padding strategy ('longest', 'max_length', True, False)
        max_length: Maximum sequence length
        pad_to_multiple_of: Pad to multiple of this value
        label_pad_token_id: Token ID to pad labels with (-100 for loss ignore)
        return_tensors: Output tensor type ('pt' for PyTorch)
        coord_pad_value: Value to pad coordinates with (default 0)

    Example:
        >>> collator = GeoDataCollatorForSeq2Seq(
        ...     tokenizer=tokenizer,
        ...     model=model,
        ...     pad_to_multiple_of=8,
        ... )
        >>> batch = collator([sample1, sample2, sample3])
        >>> batch.keys()
        dict_keys(['input_ids', 'attention_mask', 'labels',
                   'encoder_abs_positions', 'encoder_rel_positions',
                   'decoder_coordinates'])
    """

    coord_pad_value: int = 0

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        """
        Collate batch with position handling.

        Args:
            features: List of feature dictionaries from dataset
            return_tensors: Output tensor format

        Returns:
            Batched dictionary with padded sequences and positions
        """
        # Extract positions before calling parent collator
        # (parent doesn't know about position fields)
        src_abs_pos_list = []
        src_rel_pos_list = []
        tgt_coords_list = []
        has_src_abs_pos = False
        has_src_rel_pos = False
        has_tgt_coords = False

        for feature in features:
            if "src_abs_pos" in feature:
                src_abs_pos_list.append(feature.pop("src_abs_pos"))
                has_src_abs_pos = True
            if "src_rel_pos" in feature:
                src_rel_pos_list.append(feature.pop("src_rel_pos"))
                has_src_rel_pos = True
            if "tgt_coords" in feature:
                tgt_coords_list.append(feature.pop("tgt_coords"))
                has_tgt_coords = True

        # Call parent collator for standard Seq2Seq processing
        batch = super().__call__(features, return_tensors=return_tensors)

        # Get padded sequence lengths
        padded_src_len = batch["input_ids"].shape[1]
        padded_tgt_len = batch["labels"].shape[1] if "labels" in batch else None

        # Pad and add source absolute positions to batch
        if has_src_abs_pos and src_abs_pos_list:
            encoder_abs_pos = self._pad_coordinates(
                src_abs_pos_list,
                padded_src_len,
                self.coord_pad_value
            )
            batch["encoder_abs_positions"] = encoder_abs_pos

        # Pad and add source relative positions to batch
        if has_src_rel_pos and src_rel_pos_list:
            encoder_rel_pos = self._pad_coordinates(
                src_rel_pos_list,
                padded_src_len,
                self.coord_pad_value
            )
            batch["encoder_rel_positions"] = encoder_rel_pos

        # Pad and add target coordinates to batch
        if has_tgt_coords and tgt_coords_list:
            if padded_tgt_len is None:
                padded_tgt_len = max(len(coords) for coords in tgt_coords_list)
            decoder_coords = self._pad_coordinates(
                tgt_coords_list,
                padded_tgt_len,
                self.coord_pad_value
            )
            batch["decoder_coordinates"] = decoder_coords

        return batch

    def _pad_coordinates(
        self,
        coords_list: List[List[tuple]],
        target_length: int,
        pad_value: int = 0
    ) -> torch.Tensor:
        """
        Pad coordinate sequences to target length.

        Args:
            coords_list: List of coordinate sequences, each is List[(x, y, z)]
            target_length: Target sequence length after padding
            pad_value: Value to use for padding

        Returns:
            Padded coordinate tensor (batch_size, target_length, 3)
        """
        batch_size = len(coords_list)
        padded_coords = torch.full(
            (batch_size, target_length, 3),
            pad_value,
            dtype=torch.long
        )

        for i, coords in enumerate(coords_list):
            seq_len = min(len(coords), target_length)
            for j in range(seq_len):
                if isinstance(coords[j], (list, tuple)) and len(coords[j]) >= 3:
                    padded_coords[i, j, 0] = coords[j][0]  # x
                    padded_coords[i, j, 1] = coords[j][1]  # y
                    padded_coords[i, j, 2] = coords[j][2]  # z

        return padded_coords


def create_geo_data_collator(
    tokenizer: PreTrainedTokenizerBase,
    model: Optional[Any] = None,
    pad_to_multiple_of: int = 8,
    label_pad_token_id: int = -100,
    coord_pad_value: int = 0,
) -> GeoDataCollatorForSeq2Seq:
    """
    Create a GeoDataCollatorForSeq2Seq with standard settings.

    This is a convenience function for creating a data collator with
    the settings commonly used in the training pipeline.

    Args:
        tokenizer: Tokenizer for padding
        model: Model for label preparation
        pad_to_multiple_of: Pad sequences to multiple of this value
        label_pad_token_id: Token ID for padding labels (-100 to ignore in loss)
        coord_pad_value: Value for padding coordinates

    Returns:
        Configured GeoDataCollatorForSeq2Seq instance

    Example:
        >>> collator = create_geo_data_collator(tokenizer, model)
        >>> dataloader = DataLoader(dataset, collate_fn=collator)
    """
    return GeoDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=pad_to_multiple_of,
        padding=True,
        label_pad_token_id=label_pad_token_id,
        coord_pad_value=coord_pad_value,
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
