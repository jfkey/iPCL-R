#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   coordinate_utils.py
@Time    :   2025/01/14
@Author  :   Dawn Li
@Version :   2.0
@Desc    :   Coordinate computation utilities for geometry-aware trajectory generation.
             Computes absolute 3D coordinates (x, y, z) for each token in routing sequences.

             Key Features:
             - CoordinateTracker: Track absolute coordinates during tokenization
             - Coordinate computation for source and target sequences
             - Support for PUSH/POP (branch/end) operations in routing trees

             This module enables pre-computing absolute positions during tokenization
             so that training can simply load the dataset with coordinates already embedded.
"""

import logging
import re
from typing import List, Optional, Tuple, Union

from flow.utils.token_preprocessing import CoordinatePoint
from flow.utils.constants import DIRECTION_MAP, DIRECTION_TOKEN_PATTERN, COORDINATE_PATTERN


# =============================================================================
# Special Position Markers for Geometry-Aware Training
# =============================================================================

# Position marker for special tokens (BOS, EOS, SRC_END, etc.)
SPECIAL_POS = (0, 0, 0)

# Position marker for branch point (<PUSH>)
BRANCH_POS = (1, 1, 1)

# Position marker for end of branch (<POP>)
END_POS = (2, 2, 2)


# =============================================================================
# CoordinateTracker Class
# =============================================================================

class CoordinateTracker:
    """
    Track absolute coordinates during tokenization.

    This class maintains the current position and a stack for handling
    branch points (<PUSH>) and end points (<POP>) in routing trees.

    The tracker records a position for each token, ensuring that
    len(tokens) == len(positions) for the final dataset.

    Example:
        >>> tracker = CoordinateTracker()
        >>> tracker.reset(start_pos=(1000, 2000, 3))
        >>> tracker.apply_movement(dx=500, dy=0, dm=0)  # R500
        >>> tracker.push()  # <PUSH> - branch point
        >>> tracker.apply_movement(dx=0, dy=200, dm=0)  # U200
        >>> tracker.pop()   # <POP> - return to branch
        >>> positions = tracker.get_positions()
    """

    def __init__(self):
        self.current_pos = [0, 0, 0]  # (x, y, m)
        self.stack = []  # For PUSH/POP operations
        self.positions = []  # Store (x, y, m) for each token

    def reset(self, start_pos: Tuple[int, int, int] = (0, 0, 0)):
        """Reset tracker to initial position."""
        self.current_pos = list(start_pos)
        self.stack = []
        self.positions = []

    def record_position(self):
        """Record current position."""
        self.positions.append(tuple(self.current_pos))

    def apply_movement(self, dx: int = 0, dy: int = 0, dm: int = 0):
        """
        Apply relative movement and record the resulting position.

        Called after each direction token to update and record position.
        """
        self.current_pos[0] += dx
        self.current_pos[1] += dy
        self.current_pos[2] += dm
        self.record_position()

    def push(self):
        """
        Handle <PUSH> token (branch point).

        1. Save current REAL position to stack for later retrieval.
        2. Record the SPECIAL branch position marker for the dataset.
        """
        self.stack.append(self.current_pos.copy())
        self.positions.append(BRANCH_POS)

    def pop(self):
        """
        Handle <POP> token (end of branch).

        1. Restore REAL position from stack.
        2. Record the SPECIAL end position marker for the dataset.
        """
        if self.stack:
            self.current_pos = self.stack.pop()
        self.positions.append(END_POS)

    def get_positions(self) -> List[Tuple[int, int, int]]:
        """Get all recorded positions."""
        return self.positions.copy()

    def get_current_position(self) -> Tuple[int, int, int]:
        """Get current absolute position."""
        return tuple(self.current_pos)

    def record_zero_position(self):
        """Record (0, 0, 0) position for special tokens."""
        self.positions.append((0, 0, 0))

    def record_special_position(self):
        """Record special position marker for meta tokens."""
        self.positions.append(SPECIAL_POS)


def is_direction_token(token: str) -> bool:
    """
    Check if a token is a direction token (R/L/U/D/T/B followed by number).

    Args:
        token: Token string to check

    Returns:
        True if token matches direction pattern (e.g., R500, U200, T1)
    """
    if not token or not isinstance(token, str):
        return False
    return bool(DIRECTION_TOKEN_PATTERN.fullmatch(token))


def is_coordinate_token(token: str) -> bool:
    """
    Check if a token is a coordinate token (x, y, m) format.

    Args:
        token: Token string to check

    Returns:
        True if token matches coordinate pattern (e.g., (1000, 2000, 3))
    """
    if not token or not isinstance(token, str):
        return False
    return bool(COORDINATE_PATTERN.fullmatch(token.strip()))


def parse_direction_token(token: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse a direction token into (dx, dy, dz) movement.

    Args:
        token: Direction token (e.g., R500, U200, T1)

    Returns:
        Tuple of (dx, dy, dz) movement, or None if invalid
    """
    if not is_direction_token(token):
        return None

    direction_char = token[0].upper()
    distance = int(token[1:])

    base_dx, base_dy, base_dz = DIRECTION_MAP[direction_char]
    return (base_dx * distance, base_dy * distance, base_dz * distance)


def parse_coordinate_string(coord_str: str) -> Optional[CoordinatePoint]:
    """
    Parse a coordinate string into CoordinatePoint.

    Args:
        coord_str: Coordinate string (e.g., "(1000, 2000, 3)")

    Returns:
        CoordinatePoint or None if invalid
    """
    if not coord_str or not isinstance(coord_str, str):
        return None

    match = COORDINATE_PATTERN.match(coord_str.strip())
    if not match:
        return None

    x, y, m = map(int, match.groups())
    return CoordinatePoint(x, y, m)


def compute_absolute_coordinates(
    tokens: Union[str, List[str]],
    start_coord: Optional[CoordinatePoint] = None
) -> List[Tuple[int, int, int]]:
    """
    Compute absolute coordinates for each token in a sequence.

    This is the core function for Step 1 of geometry-aware trajectory generation.
    It traces the routing path and assigns (x, y, z) coordinates to each token.

    For direction tokens (R/L/U/D/T/B), coordinates accumulate from previous position.
    For special tokens (<BOS>, <PUSH>, etc.), they inherit the current position.
    For coordinate tokens from source, they are parsed directly.

    The function handles the tree structure of routing paths using a stack for
    <PUSH>/<POP> operations, which represent branching in the routing tree.

    Args:
        tokens: Token sequence as string (space-separated) or list of strings
        start_coord: Starting coordinate (defaults to origin (0, 0, 0))

    Returns:
        List of (x, y, z) tuples, one per token

    Example:
        >>> tokens = ["<BOS>", "R500", "U200", "<PUSH>", "L100", "<POP>", "D50", "<EOS>"]
        >>> coords = compute_absolute_coordinates(tokens, CoordinatePoint(1000, 2000, 3))
        >>> # coords[0] = (1000, 2000, 3)  # <BOS> at start
        >>> # coords[1] = (1500, 2000, 3)  # R500: moved +500 in x
        >>> # coords[2] = (1500, 2200, 3)  # U200: moved +200 in y
        >>> # coords[3] = (1500, 2200, 3)  # <PUSH>: saves position
        >>> # coords[4] = (1400, 2200, 3)  # L100: moved -100 in x
        >>> # coords[5] = (1500, 2200, 3)  # <POP>: restored to push point
        >>> # coords[6] = (1500, 2150, 3)  # D50: moved -50 in y
        >>> # coords[7] = (1500, 2150, 3)  # <EOS> at final position
    """
    # Handle string input
    if isinstance(tokens, str):
        tokens = tokens.split()

    # Initialize starting coordinate
    if start_coord is None:
        start_coord = CoordinatePoint(0, 0, 0)

    coordinates: List[Tuple[int, int, int]] = []
    current_coord = start_coord
    coord_stack: List[CoordinatePoint] = []  # Stack for <PUSH>/<POP> handling

    for token in tokens:
        token_stripped = token.strip()

        if is_direction_token(token_stripped):
            # Direction token: update coordinate based on movement
            movement = parse_direction_token(token_stripped)
            if movement:
                dx, dy, dz = movement
                current_coord = CoordinatePoint(
                    current_coord.x + dx,
                    current_coord.y + dy,
                    current_coord.m + dz
                )
        elif token_stripped == "<PUSH>":
            # Save current position to stack (branch point)
            coord_stack.append(current_coord)
        elif token_stripped == "<POP>":
            # Restore position from stack (return to branch point)
            if coord_stack:
                current_coord = coord_stack.pop()
        elif is_coordinate_token(token_stripped):
            # Parse absolute coordinate from source tokens
            parsed = parse_coordinate_string(token_stripped)
            if parsed:
                current_coord = parsed
        # else: special tokens (<BOS>, <EOS>, <DRIVER>, <LOAD>, etc.)
        # inherit the current position

        coordinates.append(current_coord.to_tuple())

    return coordinates


def extract_start_coordinate_from_source(
    source_tokens: Union[str, List[str]]
) -> Optional[CoordinatePoint]:
    """
    Extract the starting coordinate (driver position) from source tokens.

    The source sequence typically starts with <DRIVER> followed by a coordinate.
    This function finds the first coordinate in the source sequence.

    Args:
        source_tokens: Source token sequence

    Returns:
        Starting coordinate (driver position) or None if not found
    """
    if isinstance(source_tokens, str):
        source_tokens = source_tokens.split()

    for token in source_tokens:
        token_stripped = token.strip()
        if is_coordinate_token(token_stripped):
            return parse_coordinate_string(token_stripped)

    return None


def compute_coordinates_for_sample(
    source_tokens: Union[str, List[str]],
    target_tokens: Union[str, List[str]],
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """
    Compute coordinates for both source and target sequences of a sample.

    This is a convenience function that:
    1. Computes source coordinates starting from origin
    2. Extracts the driver position from source
    3. Computes target coordinates starting from driver position

    Args:
        source_tokens: Source token sequence (pin positions, netlist info)
        target_tokens: Target token sequence (routing directions)

    Returns:
        Tuple of (source_coords, target_coords)
    """
    # Compute source coordinates from origin
    source_coords = compute_absolute_coordinates(source_tokens)

    # Extract driver position as starting point for target
    start_coord = extract_start_coordinate_from_source(source_tokens)
    if start_coord is None:
        start_coord = CoordinatePoint(0, 0, 0)

    # Compute target coordinates from driver position
    target_coords = compute_absolute_coordinates(target_tokens, start_coord)

    return source_coords, target_coords


def normalize_coordinates(
    coordinates: List[Tuple[int, int, int]],
    scale: float = 1e-4,
    offset: Optional[Tuple[int, int, int]] = None
) -> List[Tuple[float, float, float]]:
    """
    Normalize coordinates for neural network input.

    Chip coordinates can be very large (millions of units). This function
    scales them down to a reasonable range and optionally centers them.

    Args:
        coordinates: List of (x, y, z) tuples
        scale: Scaling factor (default 1e-4 converts 10000 -> 1.0)
        offset: Optional (x, y, z) offset to subtract before scaling

    Returns:
        List of normalized (x, y, z) tuples as floats
    """
    if offset is None:
        offset = (0, 0, 0)

    normalized = []
    for x, y, z in coordinates:
        nx = (x - offset[0]) * scale
        ny = (y - offset[1]) * scale
        nz = (z - offset[2]) * scale
        normalized.append((nx, ny, nz))

    return normalized


def compute_coordinate_bounds(
    coordinates: List[Tuple[int, int, int]]
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Compute bounding box of coordinates.

    Args:
        coordinates: List of (x, y, z) tuples

    Returns:
        Tuple of (min_coord, max_coord)
    """
    if not coordinates:
        return ((0, 0, 0), (0, 0, 0))

    xs, ys, zs = zip(*coordinates)
    return (
        (min(xs), min(ys), min(zs)),
        (max(xs), max(ys), max(zs))
    )


def compute_center_offset(
    coordinates: List[Tuple[int, int, int]]
) -> Tuple[int, int, int]:
    """
    Compute center offset for normalizing coordinates.

    Args:
        coordinates: List of (x, y, z) tuples

    Returns:
        Center point as (x, y, z) tuple
    """
    min_coord, max_coord = compute_coordinate_bounds(coordinates)
    return (
        (min_coord[0] + max_coord[0]) // 2,
        (min_coord[1] + max_coord[1]) // 2,
        (min_coord[2] + max_coord[2]) // 2
    )


# =============================================================================
# Source Position Extraction (from raw data, NOT from tokens)
# =============================================================================

def extract_source_positions_from_raw_data(
    driver_str: str,
    loads: List[str],
    source_tokens: str,
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """
    Extract absolute and relative positions for source tokens from RAW DATA.

    This function extracts positions directly from the original driver/loads data,
    NOT by computing from direction tokens. Position embeddings are only applied
    to semantic tokens (<DRIVER>, <LOAD>), not direction tokens.

    Args:
        driver_str: Driver coordinate string like "(2382120, 691600, 2)"
        loads: List of load coordinate strings
        source_tokens: Source tokens string (used to find <DRIVER>/<LOAD> positions)

    Returns:
        Tuple of (abs_positions, rel_positions):
        - abs_positions: List of (x, y, m) for each token
          - <DRIVER> token gets driver's absolute position
          - <LOAD> tokens get load's absolute position
          - Other tokens get SPECIAL_POS (0, 0, 0)
        - rel_positions: List of (dx, dy, dm) for each token
          - <DRIVER> token gets (0, 0, 0)
          - <LOAD> tokens get (load - driver) relative position
          - Other tokens get SPECIAL_POS (0, 0, 0)

    Example:
        >>> driver = "(2382120, 691600, 2)"
        >>> loads = ["(2375890, 625200, 0)", "(2375890, 621600, 0)"]
        >>> tokens = "<BOS> <DRIVER> R2000000 ... <LOAD> L6000 ... <LOAD> L6000 ..."
        >>> abs_pos, rel_pos = extract_source_positions_from_raw_data(driver, loads, tokens)
    """
    # Parse driver position
    driver_coord = parse_coordinate_string(driver_str)
    if driver_coord is None:
        driver_coord = CoordinatePoint(0, 0, 0)
    driver_pos = (driver_coord.x, driver_coord.y, driver_coord.m)

    # Parse load positions
    load_positions = []
    load_relative_positions = []
    for load_str in loads:
        load_coord = parse_coordinate_string(load_str)
        if load_coord is None:
            load_coord = CoordinatePoint(0, 0, 0)
        load_pos = (load_coord.x, load_coord.y, load_coord.m)
        load_positions.append(load_pos)
        # Calculate relative position: load - driver
        rel_pos = (
            load_coord.x - driver_coord.x,
            load_coord.y - driver_coord.y,
            load_coord.m - driver_coord.m,
        )
        load_relative_positions.append(rel_pos)

    # Parse tokens and assign positions
    if isinstance(source_tokens, str):
        token_list = source_tokens.split()
    else:
        token_list = source_tokens

    abs_positions = []
    rel_positions = []
    load_idx = 0

    for token in token_list:
        token_stripped = token.strip()

        if token_stripped == "<DRIVER>":
            # Driver token: absolute position, relative = (0, 0, 0)
            abs_positions.append(driver_pos)
            rel_positions.append((0, 0, 0))
        elif token_stripped == "<LOAD>":
            # Load token: absolute position + relative position
            if load_idx < len(load_positions):
                abs_positions.append(load_positions[load_idx])
                rel_positions.append(load_relative_positions[load_idx])
                load_idx += 1
            else:
                # Fallback if more <LOAD> tokens than loads
                abs_positions.append(SPECIAL_POS)
                rel_positions.append(SPECIAL_POS)
        else:
            # Other tokens (BOS, direction tokens, SRC_END, etc.): no position
            abs_positions.append(SPECIAL_POS)
            rel_positions.append(SPECIAL_POS)

    return abs_positions, rel_positions


# =============================================================================
# Tokenization-Phase Coordinate Computation (for target tokens)
# =============================================================================

def compute_source_coordinates_from_tokens(
    source_tokens: str,
    driver_coord: Optional[CoordinatePoint] = None,
) -> List[Tuple[int, int, int]]:
    """
    DEPRECATED: Use extract_source_positions_from_raw_data instead.

    Compute absolute coordinates for source tokens during tokenization.
    This function processes source tokens (directional tokens like R500, U200)
    and computes the absolute position for each token.

    Args:
        source_tokens: Space-separated source tokens string
        driver_coord: Starting coordinate (driver position). If None, starts at (0,0,0).

    Returns:
        List of (x, y, m) tuples, one per token
    """
    import warnings
    warnings.warn(
        "compute_source_coordinates_from_tokens is deprecated. "
        "Use extract_source_positions_from_raw_data instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if isinstance(source_tokens, str):
        token_list = source_tokens.split()
    else:
        token_list = source_tokens

    tracker = CoordinateTracker()
    start_pos = (driver_coord.x, driver_coord.y, driver_coord.m) if driver_coord else (0, 0, 0)
    tracker.reset(start_pos=(0, 0, 0))  # Source starts from origin (relative to driver)

    for token in token_list:
        token_stripped = token.strip()

        if is_direction_token(token_stripped):
            # Direction token: update coordinate based on movement
            movement = parse_direction_token(token_stripped)
            if movement:
                dx, dy, dz = movement
                tracker.apply_movement(dx=dx, dy=dy, dm=dz)
        elif token_stripped == "<PUSH>":
            tracker.push()
        elif token_stripped == "<POP>":
            tracker.pop()
        else:
            # Special tokens: record special position marker
            tracker.record_special_position()

    return tracker.get_positions()


def compute_target_coordinates_from_tokens(
    target_tokens: str,
    driver_coord: Optional[CoordinatePoint] = None,
) -> List[Tuple[int, int, int]]:
    """
    Compute absolute coordinates for target tokens during tokenization.

    Target tokens represent the routing path and need to track absolute
    positions starting from the driver location.

    Args:
        target_tokens: Space-separated target tokens string
        driver_coord: Starting coordinate (driver position). If None, starts at (0,0,0).

    Returns:
        List of (x, y, m) tuples, one per token

    Example:
        >>> tokens = "R500 U200 <PUSH> L100 <POP> D50 <EOS>"
        >>> coords = compute_target_coordinates_from_tokens(tokens, CoordinatePoint(1000, 2000, 3))
    """
    if isinstance(target_tokens, str):
        token_list = target_tokens.split()
    else:
        token_list = target_tokens

    tracker = CoordinateTracker()
    start_pos = (driver_coord.x, driver_coord.y, driver_coord.m) if driver_coord else (0, 0, 0)
    tracker.reset(start_pos=start_pos)

    for token in token_list:
        token_stripped = token.strip()

        if is_direction_token(token_stripped):
            # Direction token: update coordinate based on movement
            movement = parse_direction_token(token_stripped)
            if movement:
                dx, dy, dz = movement
                tracker.apply_movement(dx=dx, dy=dy, dm=dz)
        elif token_stripped == "<PUSH>":
            tracker.push()
        elif token_stripped == "<POP>":
            tracker.pop()
        elif token_stripped in ("<EOS>", "<PAD>"):
            # End/pad tokens: record special position
            tracker.record_special_position()
        else:
            # Other tokens: inherit current position
            tracker.record_position()

    return tracker.get_positions()


def compute_coordinates_for_tokenized_sample(
    source_tokens: str,
    target_tokens: str,
    driver_str: Optional[str] = None,
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """
    Compute coordinates for both source and target token sequences.

    This is the main entry point for computing coordinates during tokenization.
    It extracts the driver position and computes absolute coordinates for
    both source and target sequences.

    Args:
        source_tokens: Space-separated source tokens string
        target_tokens: Space-separated target tokens string
        driver_str: Driver coordinate string like "(1000, 2000, 3)". If None,
                    attempts to extract from source tokens.

    Returns:
        Tuple of (source_coords, target_coords)

    Example:
        >>> src = "<BOS> <DRIVER> R500 U200 <LOAD> L100 <SRC_END>"
        >>> tgt = "R500 U200 <PUSH> L100 <POP> <EOS>"
        >>> src_coords, tgt_coords = compute_coordinates_for_tokenized_sample(
        ...     src, tgt, driver_str="(1000, 2000, 3)"
        ... )
    """
    # Parse driver coordinate
    driver_coord = None
    if driver_str:
        driver_coord = parse_coordinate_string(driver_str)
    else:
        # Try to extract from source
        driver_coord = extract_start_coordinate_from_source(source_tokens)

    if driver_coord is None:
        driver_coord = CoordinatePoint(0, 0, 0)

    # Compute coordinates for source and target
    source_coords = compute_source_coordinates_from_tokens(source_tokens, driver_coord)
    target_coords = compute_target_coordinates_from_tokens(target_tokens, driver_coord)

    return source_coords, target_coords


def validate_coordinate_alignment(
    tokens: List[str],
    coordinates: List[Tuple[int, int, int]],
) -> bool:
    """
    Validate that tokens and coordinates have matching lengths.

    This is a sanity check to ensure coordinate computation is correct.

    Args:
        tokens: List of token strings
        coordinates: List of (x, y, m) tuples

    Returns:
        True if lengths match, False otherwise

    Raises:
        ValueError: If lengths don't match and logging level allows
    """
    if len(tokens) != len(coordinates):
        logging.error(
            f"Token-coordinate mismatch: {len(tokens)} tokens vs {len(coordinates)} coordinates"
        )
        return False
    return True
