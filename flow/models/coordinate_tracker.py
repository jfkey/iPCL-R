#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   coordinate_tracker.py
@Time    :   2025/01/29
@Author  :   Claude Code
@Version :   1.0
@Desc    :   Real-time coordinate tracking for inference-time generation.

             This module provides a CoordinateTracker that parses generated
             tokens in real-time and maintains the current 3D position during
             autoregressive decoding.

             This solves the chicken-and-egg problem in LARA inference:
             - LARA needs coordinates to generate tokens
             - But coordinates are derived from tokens

             Solution: Parse each generated token immediately and update position.
"""

import re
from typing import List, Tuple, Optional

import torch


class InferenceCoordinateTracker:
    """
    Real-time coordinate tracker for inference-time token generation.

    Maintains the current 3D position and branch stack while tokens are
    being generated autoregressively. Each time a new token is generated,
    it is parsed and the position is updated.

    This tracker implements the same logic as the training-time
    CoordinateTracker (from tokenization/coordinate_utils.py) but operates
    incrementally during generation.

    Key Features:
    - Incremental updates: add one token at a time
    - <PUSH>/<POP> stack management with spatial inheritance
    - Direction token parsing (R/L/U/D/T/B + distance)
    - Returns history of coordinates compatible with LARA

    Example:
        >>> tracker = InferenceCoordinateTracker(driver_pos=(1000, 2000, 3))
        >>> # Generate tokens one by one
        >>> tracker.update('R500')      # Move right
        >>> tracker.update('<PUSH>')    # Branch point
        >>> tracker.update('U200')      # Move up
        >>> tracker.update('<POP>')     # Return to branch
        >>> coords_tensor = tracker.get_coords_tensor()  # (seq_len, 3)
    """

    # Direction mapping (same as in tokenization)
    DIRECTION_MAP = {
        'R': (1, 0, 0),   # Right (+x)
        'L': (-1, 0, 0),  # Left (-x)
        'U': (0, 1, 0),   # Up (+y)
        'D': (0, -1, 0),  # Down (-y)
        'T': (0, 0, 1),   # Top (layer up, +z)
        'B': (0, 0, -1),  # Bottom (layer down, -z)
    }

    # Regex pattern for direction tokens: R500, U200, T2, etc.
    DIRECTION_PATTERN = re.compile(r'^([RLUDTB])(\d+)$')

    def __init__(self, driver_pos: Tuple[int, int, int] = (0, 0, 0)):
        """
        Initialize tracker at driver position.

        Args:
            driver_pos: Starting position (x, y, z) - typically the driver coordinate
        """
        self.current_pos = list(driver_pos)  # [x, y, z]
        self.stack = []  # Stack for <PUSH>/<POP> operations
        self.history_coords = [driver_pos]  # History of all coordinates

        # Store driver position for reference
        self.driver_pos = driver_pos

    def reset(self, driver_pos: Tuple[int, int, int] = (0, 0, 0)):
        """Reset tracker to a new starting position."""
        self.current_pos = list(driver_pos)
        self.stack = []
        self.history_coords = [driver_pos]
        self.driver_pos = driver_pos

    def update(self, token: str) -> Tuple[int, int, int]:
        """
        Update position based on newly generated token.

        This is called immediately after each token is generated to
        maintain real-time position tracking.

        Args:
            token: Newly generated token (e.g., 'D30000', '<PUSH>', 'T2')

        Returns:
            Coordinate for this token (x, y, z)
        """
        token_stripped = token.strip()

        if token_stripped == '<PUSH>':
            # Save current position to stack
            self.stack.append(tuple(self.current_pos))
            # KEY: Inherit current position (not a dummy marker)
            coord = tuple(self.current_pos)

        elif token_stripped == '<POP>':
            # Restore position from stack
            if self.stack:
                parent_pos = self.stack.pop()
                self.current_pos = list(parent_pos)
                # KEY: Inherit parent position (not a dummy marker)
                coord = parent_pos
            else:
                # Fallback: no parent to return to
                coord = tuple(self.current_pos)

        elif self._is_direction_token(token_stripped):
            # Parse and apply directional movement
            delta = self._parse_direction_token(token_stripped)
            if delta:
                self.current_pos[0] += delta[0]
                self.current_pos[1] += delta[1]
                self.current_pos[2] += delta[2]
            coord = tuple(self.current_pos)

        else:
            # Unknown/special token: inherit current position
            coord = tuple(self.current_pos)

        # Record to history
        self.history_coords.append(coord)
        return coord

    def _is_direction_token(self, token: str) -> bool:
        """Check if token is a direction token (R500, U200, etc.)."""
        return bool(self.DIRECTION_PATTERN.match(token))

    def _parse_direction_token(self, token: str) -> Optional[Tuple[int, int, int]]:
        """
        Parse direction token into (dx, dy, dz) movement.

        Args:
            token: Direction token (e.g., 'R500', 'U200', 'T2')

        Returns:
            (dx, dy, dz) movement tuple, or None if invalid
        """
        match = self.DIRECTION_PATTERN.match(token)
        if not match:
            return None

        direction_char = match.group(1)
        try:
            distance = int(match.group(2))
        except ValueError:
            return None

        base_dx, base_dy, base_dz = self.DIRECTION_MAP.get(direction_char, (0, 0, 0))
        return (base_dx * distance, base_dy * distance, base_dz * distance)

    def get_current_position(self) -> Tuple[int, int, int]:
        """Get current absolute position."""
        return tuple(self.current_pos)

    def get_history_coords(self) -> List[Tuple[int, int, int]]:
        """Get list of all historical coordinates."""
        return self.history_coords.copy()

    def get_coords_tensor(self, device: torch.device = None) -> torch.Tensor:
        """
        Get coordinates as PyTorch tensor for LARA.

        Args:
            device: Target device (cpu, cuda, etc.). If None, uses cpu.

        Returns:
            Tensor of shape (seq_len, 3) with all historical coordinates
        """
        if device is None:
            device = torch.device('cpu')
        return torch.tensor(self.history_coords, dtype=torch.long, device=device)

    def get_batch_coords_tensor(
        self,
        batch_size: int = 1,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Get coordinates as batched tensor.

        Useful for generation where batch_size=1 but model expects (batch, seq, 3).

        Args:
            batch_size: Batch dimension size
            device: Target device

        Returns:
            Tensor of shape (batch_size, seq_len, 3)
        """
        coords = self.get_coords_tensor(device)  # (seq_len, 3)
        return coords.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, seq_len, 3)

    def __len__(self) -> int:
        """Return number of tokens processed (length of history)."""
        return len(self.history_coords)

    def __repr__(self) -> str:
        return (
            f"InferenceCoordinateTracker("
            f"current_pos={self.current_pos}, "
            f"history_len={len(self.history_coords)}, "
            f"stack_depth={len(self.stack)})"
        )


def generate_with_coordinate_tracking(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    encoder_abs_positions: torch.Tensor,
    encoder_rel_positions: Optional[torch.Tensor] = None,
    driver_pos: Tuple[int, int, int] = (0, 0, 0),
    max_length: int = 512,
    **generate_kwargs
) -> Tuple[torch.Tensor, List[Tuple[int, int, int]]]:
    """
    Generate tokens with real-time coordinate tracking for LARA.

    This function wraps the model's generate() method and maintains
    a coordinate tracker that updates after each generated token.

    Args:
        model: GeoT5GemmaForConditionalGeneration model
        tokenizer: Tokenizer for decoding token IDs
        input_ids: Encoder input token IDs (batch, src_len)
        encoder_abs_positions: Encoder absolute positions (batch, src_len, 3)
        encoder_rel_positions: Encoder relative positions (batch, src_len, 3)
        driver_pos: Starting position for decoder path
        max_length: Maximum generation length
        **generate_kwargs: Additional arguments for model.generate()

    Returns:
        Tuple of (generated_ids, decoder_coordinates)
        - generated_ids: (batch, gen_len) tensor
        - decoder_coordinates: List of (x, y, z) coordinates

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained(...)
        >>> model = GeoT5GemmaForConditionalGeneration(...)
        >>> driver_pos = (2382120, 691600, 2)
        >>> generated_ids, coords = generate_with_coordinate_tracking(
        ...     model, tokenizer, input_ids, encoder_abs_pos, driver_pos=driver_pos
        ... )
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]

    # Initialize coordinate tracker
    tracker = InferenceCoordinateTracker(driver_pos=driver_pos)

    # Prepare encoder outputs (compute once, reuse for all decoder steps)
    # Import here to avoid circular import
    from .geo_t5gemma import GeoT5GemmaEncoder

    with torch.no_grad():
        encoder = model.get_encoder()
        if isinstance(encoder, GeoT5GemmaEncoder):
            encoder_outputs = encoder(
                input_ids=input_ids,
                coordinates=encoder_abs_positions,
            )
        else:
            # Standard encoder
            encoder_outputs = encoder(input_ids=input_ids)

    # Initialize decoder input with BOS token
    decoder_input_ids = torch.tensor(
        [[model.config.decoder_start_token_id or model.config.bos_token_id]],
        device=device
    )

    # Autoregressive generation loop
    generated_tokens = []

    for step in range(max_length):
        # Get current coordinates from tracker
        decoder_coords = tracker.get_batch_coords_tensor(
            batch_size=batch_size, device=device
        )

        # Forward pass
        with torch.no_grad():
            outputs = model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                decoder_coordinates=decoder_coords,
                encoder_abs_positions=encoder_abs_positions,
            )

        # Get next token logits
        next_token_logits = outputs.logits[:, -1, :]

        # Sample or greedy decode
        if generate_kwargs.get('do_sample', False):
            # Sampling
            probs = torch.softmax(next_token_logits / generate_kwargs.get('temperature', 1.0), dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Decode token to string
        next_token_str = tokenizer.decode(next_token_id[0], skip_special_tokens=False)

        # Update coordinate tracker
        tracker.update(next_token_str)

        # Append to generated sequence
        generated_tokens.append(next_token_id)
        decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=1)

        # Check for EOS
        if next_token_id.item() == model.config.eos_token_id:
            break

    # Concatenate all generated tokens
    if generated_tokens:
        generated_ids = torch.cat(generated_tokens, dim=1)
    else:
        generated_ids = decoder_input_ids[:, 1:]  # Remove BOS

    return generated_ids, tracker.get_history_coords()
