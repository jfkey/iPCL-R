#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   coordinate_utils.py
@Time    :   2025/01/14
@Author  :   Dawn Li
@Version :   1.0
@Desc    :   Coordinate computation utilities for geometry-aware trajectory generation.
             Computes absolute 3D coordinates (x, y, z) for each token in routing sequences.
             Step 1 of the Geometry-Aware Trajectory Generation upgrade.
"""

import re
from typing import List, Optional, Tuple, Union

from flow.utils.token_preprocessing import CoordinatePoint
from flow.utils.constants import DIRECTION_MAP, DIRECTION_TOKEN_PATTERN, COORDINATE_PATTERN


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
