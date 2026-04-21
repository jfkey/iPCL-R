#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   bpe_merger.py
@Time    :   2026/04/04
@Author  :   Junfeng Liu
@Desc    :   BPE merger that learns and applies byte-pair merges on top of
             DecimalWordLevel token sequences. Special tokens act as merge
             boundaries and are never merged.

Optimized with:
  - Token -> int mapping for faster hashing/comparison
  - Incremental pair counting (only update affected sequences per merge step)
  - Parallel initial pair counting via multiprocessing
"""

import json
import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

from flow.tokenization.coordinate_utils import parse_direction_token


# ---------------------------------------------------------------------------
# Module-level helpers for multiprocessing (must be top-level for pickling)
# ---------------------------------------------------------------------------
_BPE_SHARED_SEQS = None
_BPE_SHARED_SPECIAL = None


def _bpe_count_chunk(args):
    """Count pairs and build per-pair sequence lists for a chunk."""
    start, end = args
    seqs = _BPE_SHARED_SEQS
    special_ids = _BPE_SHARED_SPECIAL
    counts = {}
    pair_seqs = {}
    for seq_idx in range(start, end):
        seq = seqs[seq_idx]
        prev = -1
        for tok in seq:
            if tok in special_ids:
                prev = -1
                continue
            if prev >= 0:
                pair = (prev, tok)
                counts[pair] = counts.get(pair, 0) + 1
                if pair in pair_seqs:
                    pair_seqs[pair].append(seq_idx)
                else:
                    pair_seqs[pair] = [seq_idx]
            prev = tok
    return counts, pair_seqs


def _bpe_seq_pairs(seq, special_ids):
    """Return {pair: count} for adjacent non-special pairs in *seq*."""
    counts = {}
    prev = -1
    for tok in seq:
        if tok in special_ids:
            prev = -1
            continue
        if prev >= 0:
            pair = (prev, tok)
            counts[pair] = counts.get(pair, 0) + 1
        prev = tok
    return counts


def _bpe_apply_merge(seq, a, b, merged_id, special_ids):
    """Apply a single merge (a,b)->merged_id to an int sequence."""
    n = len(seq)
    if n < 2:
        return seq
    new_seq = []
    i = 0
    while i < n:
        if (
            i < n - 1
            and seq[i] == a
            and seq[i + 1] == b
            and a not in special_ids
            and b not in special_ids
        ):
            new_seq.append(merged_id)
            i += 2
        else:
            new_seq.append(seq[i])
            i += 1
    return new_seq


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

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(self, sequences: List[List[str]]) -> None:
        """Learn BPE merge rules from token sequences.

        Optimized with int-mapped tokens, incremental pair counting,
        and parallel initial counting.

        Modifies *sequences* in-place to contain merged tokens.
        """
        logging.info(
            f"Learning BPE merges: up to {self.num_merges} merges "
            f"from {len(sequences)} sequences"
        )
        t0 = time.time()

        # --- Step 1: Token -> int mapping ---
        token2id: Dict[str, int] = {}
        id2token: List[str] = []

        def get_id(tok: str) -> int:
            tid = token2id.get(tok)
            if tid is None:
                tid = len(id2token)
                token2id[tok] = tid
                id2token.append(tok)
            return tid

        special_ids = frozenset(get_id(st) for st in self.special_tokens)

        # --- Step 2: Convert sequences to int lists ---
        int_seqs: List[List[int]] = [
            [get_id(tok) for tok in seq] for seq in sequences
        ]
        logging.info(
            f"Token mapping: {len(id2token)} unique tokens ({time.time()-t0:.1f}s)"
        )

        # --- Step 3: Parallel initial pair counting + inverted index ---
        t1 = time.time()
        pair_counts, pair_to_seqs = self._initial_count(int_seqs, special_ids)
        logging.info(
            f"Initial counting: {len(pair_counts)} unique pairs "
            f"({time.time()-t1:.1f}s)"
        )

        # --- Step 4: Incremental merge loop ---
        t2 = time.time()
        for step in range(self.num_merges):
            if not pair_counts:
                logging.info(f"No more pairs to merge after {step} merges")
                break

            # Find best pair
            best_pair = max(pair_counts, key=pair_counts.__getitem__)
            count = pair_counts[best_pair]

            if count < 2:
                logging.info(
                    f"Stopping BPE: best pair count {count} < 2 "
                    f"after {step} merges"
                )
                break

            a, b = best_pair
            a_str, b_str = id2token[a], id2token[b]
            merged_str = f"{a_str}{self.MERGE_SEPARATOR}{b_str}"
            merged_id = get_id(merged_str)
            self.merges.append((a_str, b_str))

            # Pop best pair from global state
            affected = pair_to_seqs.pop(best_pair, set())
            del pair_counts[best_pair]

            # Incrementally update only affected sequences
            for seq_idx in affected:
                seq = int_seqs[seq_idx]

                # Pair snapshot before merge
                old_pc = _bpe_seq_pairs(seq, special_ids)

                # Apply merge
                new_seq = _bpe_apply_merge(seq, a, b, merged_id, special_ids)
                int_seqs[seq_idx] = new_seq

                # Pair snapshot after merge
                new_pc = _bpe_seq_pairs(new_seq, special_ids)

                # Diff → update global pair_counts and inverted index
                all_pairs = set(old_pc)
                all_pairs.update(new_pc)
                for p in all_pairs:
                    if p == best_pair:
                        continue
                    old_c = old_pc.get(p, 0)
                    new_c = new_pc.get(p, 0)
                    if old_c == new_c:
                        continue

                    new_global = pair_counts.get(p, 0) + (new_c - old_c)
                    if new_global <= 0:
                        pair_counts.pop(p, None)
                    else:
                        pair_counts[p] = new_global

                    # Inverted index bookkeeping
                    if new_c > 0 and old_c == 0:
                        if p in pair_to_seqs:
                            pair_to_seqs[p].add(seq_idx)
                        else:
                            pair_to_seqs[p] = {seq_idx}
                    elif new_c == 0 and old_c > 0:
                        s = pair_to_seqs.get(p)
                        if s is not None:
                            s.discard(seq_idx)
                            if not s:
                                del pair_to_seqs[p]

            if (step + 1) % 50 == 0 or step == 0:
                elapsed = time.time() - t2
                logging.info(
                    f"  BPE merge {step+1}/{self.num_merges}: "
                    f"{a_str} + {b_str} -> {merged_str} "
                    f"(count={count}, affected={len(affected)}, "
                    f"elapsed={elapsed:.1f}s)"
                )

        # --- Step 5: Write back to string sequences (in-place) ---
        for i, seq in enumerate(int_seqs):
            sequences[i] = [id2token[tid] for tid in seq]

        total = time.time() - t0
        logging.info(f"Learned {len(self.merges)} BPE merges in {total:.1f}s")
        self._build_token_deltas()

    @staticmethod
    def _initial_count(int_seqs, special_ids):
        """Parallel initial pair counting + inverted index construction."""
        global _BPE_SHARED_SEQS, _BPE_SHARED_SPECIAL
        _BPE_SHARED_SEQS = int_seqs
        _BPE_SHARED_SPECIAL = special_ids

        n_workers = min(os.cpu_count() or 1, 32)
        chunk_size = max(1, (len(int_seqs) + n_workers - 1) // n_workers)
        chunk_args = [
            (i, min(i + chunk_size, len(int_seqs)))
            for i in range(0, len(int_seqs), chunk_size)
        ]

        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            results = pool.map(_bpe_count_chunk, chunk_args)

        _BPE_SHARED_SEQS = None
        _BPE_SHARED_SPECIAL = None

        # Merge worker results
        pair_counts: Dict[tuple, int] = {}
        pair_to_seqs: Dict[tuple, set] = {}
        for chunk_counts, chunk_ps in results:
            for pair, count in chunk_counts.items():
                pair_counts[pair] = pair_counts.get(pair, 0) + count
            for pair, idx_list in chunk_ps.items():
                if pair in pair_to_seqs:
                    pair_to_seqs[pair].update(idx_list)
                else:
                    pair_to_seqs[pair] = set(idx_list)
        return pair_counts, pair_to_seqs

    # ------------------------------------------------------------------
    # Applying merges (inference time)
    # ------------------------------------------------------------------

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
