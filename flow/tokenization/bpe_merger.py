#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   bpe_merger.py
@Time    :   2026/04/04
@Author  :   Junfeng Liu
@Desc    :   BPE merger that learns and applies byte-pair merges on top of
             DecimalWordLevel token sequences. Special tokens act as merge
             boundaries and are never merged.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

from flow.tokenization.coordinate_utils import parse_direction_token


class BPEMerger:
    """Learn and apply BPE merges on top of DecimalWordLevel token sequences.

    Special tokens act as merge boundaries and are never merged.
    Each merged token's spatial delta is the sum of its constituent base tokens' deltas.
    """

    MERGE_SEPARATOR = "_"

    def __init__(self, special_tokens: Set[str], num_merges: int = 500):
        self.special_tokens = set(special_tokens)
        self.num_merges = num_merges
        self.merges: List[Tuple[str, str]] = []
        self.token_deltas: Dict[str, Tuple[int, int, int]] = {}

    def learn(self, sequences: List[List[str]]) -> None:
        """Learn BPE merge rules from token sequences.

        Modifies sequences in-place by applying each merge as it is learned.

        Args:
            sequences: List of token sequences (lists of strings).
                       Modified in-place to contain merged tokens.
        """
        logging.info(
            f"Learning BPE merges: up to {self.num_merges} merges "
            f"from {len(sequences)} sequences"
        )

        for step in range(self.num_merges):
            # Count adjacent non-special pairs
            pair_counts = Counter()
            for seq in sequences:
                prev = None
                for tok in seq:
                    if tok in self.special_tokens:
                        prev = None
                        continue
                    if prev is not None:
                        pair_counts[(prev, tok)] += 1
                    prev = tok

            if not pair_counts:
                logging.info(f"No more pairs to merge after {step} merges")
                break

            best_pair, count = pair_counts.most_common(1)[0]
            if count < 2:
                logging.info(
                    f"Stopping BPE: best pair count {count} < 2 after {step} merges"
                )
                break

            merged = f"{best_pair[0]}{self.MERGE_SEPARATOR}{best_pair[1]}"
            self.merges.append(best_pair)

            # Apply merge to all sequences
            for i in range(len(sequences)):
                sequences[i] = self._apply_single_merge(
                    sequences[i], best_pair, merged
                )

            if (step + 1) % 100 == 0:
                logging.info(
                    f"  BPE merge {step + 1}/{self.num_merges}: "
                    f"{best_pair[0]} + {best_pair[1]} -> {merged} (count={count})"
                )

        logging.info(f"Learned {len(self.merges)} BPE merges")
        self._build_token_deltas()

    def _apply_single_merge(
        self, seq: List[str], pair: Tuple[str, str], merged: str
    ) -> List[str]:
        """Apply one merge rule to a sequence, skipping special tokens."""
        if len(seq) < 2:
            return seq
        new_seq = []
        i = 0
        while i < len(seq):
            if (
                i < len(seq) - 1
                and seq[i] == pair[0]
                and seq[i + 1] == pair[1]
                and seq[i] not in self.special_tokens
                and seq[i + 1] not in self.special_tokens
            ):
                new_seq.append(merged)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        return new_seq

    def apply(self, seq: List[str]) -> List[str]:
        """Apply all learned merges to a token sequence.

        Args:
            seq: List of base (or partially merged) tokens.

        Returns:
            New list with all merges applied.
        """
        result = list(seq)
        for pair in self.merges:
            merged = f"{pair[0]}{self.MERGE_SEPARATOR}{pair[1]}"
            result = self._apply_single_merge(result, pair, merged)
        return result

    def expand_token(self, token: str) -> List[str]:
        """Expand a merged token back to its base tokens.

        E.g. ``"R200_D300_T2"`` -> ``["R200", "D300", "T2"]``.
        Special tokens and base tokens are returned as-is.
        """
        if self.MERGE_SEPARATOR in token and token not in self.special_tokens:
            return token.split(self.MERGE_SEPARATOR)
        return [token]

    def expand_sequence(self, seq: List[str]) -> List[str]:
        """Expand all merged tokens in a sequence back to base tokens."""
        result = []
        for tok in seq:
            result.extend(self.expand_token(tok))
        return result

    # ------------------------------------------------------------------
    # Coordinate merging
    # ------------------------------------------------------------------

    def merge_with_coordinates(
        self,
        base_tokens: List[str],
        *coord_lists: List[Tuple[int, int, int]],
    ):
        """Apply merges and merge multiple coordinate lists simultaneously.

        For each merged token the coordinate chosen is that of its **last**
        constituent base token (i.e. the position after executing the full
        merged movement).

        Args:
            base_tokens: Token sequence in base DecimalWordLevel form.
            *coord_lists: One or more coordinate lists aligned with *base_tokens*.

        Returns:
            Tuple of ``(merged_tokens, merged_coords_1, merged_coords_2, ...)``.
        """
        for cl in coord_lists:
            assert len(base_tokens) == len(cl), (
                f"Token/coord length mismatch: {len(base_tokens)} vs {len(cl)}"
            )

        merged_tokens = self.apply(list(base_tokens))

        results = [[] for _ in coord_lists]
        base_idx = 0
        for mtok in merged_tokens:
            n_base = len(self.expand_token(mtok))
            last_idx = base_idx + n_base - 1
            for r, coords in zip(results, coord_lists):
                r.append(coords[last_idx])
            base_idx += n_base

        assert base_idx == len(base_tokens), (
            f"Base index {base_idx} != base_tokens length {len(base_tokens)}"
        )

        return (merged_tokens, *results)

    # ------------------------------------------------------------------
    # Token delta computation
    # ------------------------------------------------------------------

    def _build_token_deltas(self) -> None:
        """Compute the cumulative (dx, dy, dz) delta for every merged token."""
        self.token_deltas = {}
        seen = set()
        for pair in self.merges:
            merged = f"{pair[0]}{self.MERGE_SEPARATOR}{pair[1]}"
            if merged in seen:
                continue
            seen.add(merged)
            base_tokens = self.expand_token(merged)
            dx, dy, dz = 0, 0, 0
            for bt in base_tokens:
                delta = parse_direction_token(bt)
                if delta:
                    dx += delta[0]
                    dy += delta[1]
                    dz += delta[2]
            self.token_deltas[merged] = (dx, dy, dz)

    def get_token_delta(self, token: str) -> Tuple[int, int, int]:
        """Return the spatial delta (dx, dy, dz) for a (possibly merged) token."""
        if token in self.token_deltas:
            return self.token_deltas[token]
        # Base direction token (e.g. R200, D300)
        delta = parse_direction_token(token)
        if delta:
            return delta
        # Any merged token not in cache: sum deltas of its base tokens
        if self.MERGE_SEPARATOR in token and token not in self.special_tokens:
            dx, dy, dz = 0, 0, 0
            for bt in token.split(self.MERGE_SEPARATOR):
                d = parse_direction_token(bt)
                if d:
                    dx += d[0]; dy += d[1]; dz += d[2]
            return (dx, dy, dz)
        # Special or other non-direction token
        return (0, 0, 0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save merge rules and deltas to JSON."""
        data = {
            "merges": self.merges,
            "token_deltas": {k: list(v) for k, v in self.token_deltas.items()},
            "special_tokens": sorted(self.special_tokens),
            "num_merges": self.num_merges,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"BPE merger saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BPEMerger":
        """Load merge rules from JSON."""
        with open(path) as f:
            data = json.load(f)
        merger = cls(
            special_tokens=set(data["special_tokens"]),
            num_merges=data["num_merges"],
        )
        merger.merges = [tuple(m) for m in data["merges"]]
        merger.token_deltas = {k: tuple(v) for k, v in data["token_deltas"].items()}
        return merger
