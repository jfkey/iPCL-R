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
    Data collator for sequence-to-sequence models with coordinate support.

    Extends DataCollatorForSeq2Seq to handle 3D coordinate tensors alongside
    token sequences. Coordinates are padded to match the padded sequence length.

    Expected input features:
    - input_ids: Encoder input token IDs
    - attention_mask: Encoder attention mask
    - labels: Decoder target token IDs
    - source_coords: List of (x, y, z) tuples for encoder tokens
    - target_coords: List of (x, y, z) tuples for decoder tokens

    Output batch includes:
    - input_ids, attention_mask, labels (standard Seq2Seq)
    - encoder_coordinates: Padded 3D coords (batch, src_len, 3)
    - decoder_coordinates: Padded 3D coords (batch, tgt_len, 3)

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
                   'encoder_coordinates', 'decoder_coordinates'])
    """

    coord_pad_value: int = 0

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        """
        Collate batch with coordinate handling.

        Args:
            features: List of feature dictionaries from dataset
            return_tensors: Output tensor format

        Returns:
            Batched dictionary with padded sequences and coordinates
        """
        # Extract coordinates before calling parent collator
        # (parent doesn't know about coordinate fields)
        source_coords_list = []
        target_coords_list = []
        has_source_coords = False
        has_target_coords = False

        for feature in features:
            if "source_coords" in feature:
                source_coords_list.append(feature.pop("source_coords"))
                has_source_coords = True
            if "target_coords" in feature:
                target_coords_list.append(feature.pop("target_coords"))
                has_target_coords = True

        # Call parent collator for standard Seq2Seq processing
        batch = super().__call__(features, return_tensors=return_tensors)

        # Pad and add coordinates to batch
        if has_source_coords and source_coords_list:
            # Get padded sequence length from input_ids
            padded_src_len = batch["input_ids"].shape[1]
            encoder_coords = self._pad_coordinates(
                source_coords_list,
                padded_src_len,
                self.coord_pad_value
            )
            batch["encoder_coordinates"] = encoder_coords

        if has_target_coords and target_coords_list:
            # Get padded sequence length from labels
            if "labels" in batch:
                padded_tgt_len = batch["labels"].shape[1]
            else:
                # Fallback: compute from target_coords
                padded_tgt_len = max(len(coords) for coords in target_coords_list)
            decoder_coords = self._pad_coordinates(
                target_coords_list,
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
    num_workers: int = 16,
):
    """
    Add coordinate columns to a HuggingFace dataset.

    This function processes a dataset to compute and add source_coords
    and target_coords columns based on the token sequences.

    Args:
        dataset: HuggingFace Dataset with source_tokens and target_tokens
        source_col: Name of source tokens column
        target_col: Name of target tokens column
        num_workers: Number of parallel workers for processing

    Returns:
        Dataset with added source_coords and target_coords columns

    Example:
        >>> dataset = load_from_disk("token_dataset")
        >>> dataset = add_coordinates_to_dataset(dataset)
        >>> # Now dataset has 'source_coords' and 'target_coords' columns
    """
    from flow.tokenization.coordinate_utils import (
        compute_coordinates_for_sample,
    )

    def compute_coords(batch):
        """Compute coordinates for a batch of samples."""
        source_coords_batch = []
        target_coords_batch = []

        for source_tokens, target_tokens in zip(
            batch[source_col], batch[target_col]
        ):
            source_coords, target_coords = compute_coordinates_for_sample(
                source_tokens, target_tokens
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
