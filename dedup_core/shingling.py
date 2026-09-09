"""
High-performance k-shingle tokenization and 64-bit hashing.
"""

from __future__ import annotations

import re
import string
from enum import Enum
from typing import Set, List


class ShingleType(str, Enum):
    WORD = "word"
    CHAR = "char"


# FNV-1a 64-bit constants
_FNV_OFFSET_BASIS = 0xcbf29ce484222325
_FNV_PRIME = 0x100000001b3
_MASK64 = 0xffffffffffffffff


def fnv1a_64(text: str) -> int:
    """Calculates 64-bit FNV-1a hash of a UTF-8 string."""
    hash_val = _FNV_OFFSET_BASIS
    for char in text.encode("utf-8"):
        hash_val ^= char
        hash_val = (hash_val * _FNV_PRIME) & _MASK64
    return hash_val


class Shingler:
    """Generates sets of hashed k-shingles from text documents."""

    def __init__(self, k: int = 5, shingle_type: ShingleType = ShingleType.WORD):
        if k < 1:
            raise ValueError(f"Shingle size k must be >= 1, received {k}")
        self.k = k
        self.shingle_type = shingle_type
        self._punct_table = str.maketrans("", "", string.punctuation)

    def _normalize_text(self, text: str) -> str:
        """Lowercases and cleans whitespace."""
        return " ".join(text.lower().translate(self._punct_table).split())

    def get_shingle_hashes(self, text: str) -> Set[int]:
        """Extracts unique 64-bit integer hashes for all k-shingles in text."""
        cleaned = self._normalize_text(text)
        if not cleaned:
            return set()

        hashes: Set[int] = set()

        if self.shingle_type == ShingleType.WORD:
            tokens: List[str] = cleaned.split()
            if len(tokens) < self.k:
                # If document has fewer tokens than k, use the entire document as a single shingle
                return {fnv1a_64(" ".join(tokens))}

            for i in range(len(tokens) - self.k + 1):
                shingle = " ".join(tokens[i : i + self.k])
                hashes.add(fnv1a_64(shingle))

        elif self.shingle_type == ShingleType.CHAR:
            compact = cleaned.replace(" ", "")
            if len(compact) < self.k:
                return {fnv1a_64(compact)}

            for i in range(len(compact) - self.k + 1):
                shingle = compact[i : i + self.k]
                hashes.add(fnv1a_64(shingle))

        return hashes
