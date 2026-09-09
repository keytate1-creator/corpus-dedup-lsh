"""
MinHash signature generation using universal 64-bit hashing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Set, List, Tuple

# Large Mersenne prime 2^61 - 1
_MERSENNE_PRIME_61 = (1 << 61) - 1
_MASK64 = 0xffffffffffffffff


@dataclass(frozen=True)
class MinHashSignature:
    """Represents the fixed-length MinHash signature of a document."""
    doc_id: str
    signature: Tuple[int, ...]

    def estimate_jaccard(self, other: MinHashSignature) -> float:
        """Estimates Jaccard similarity by comparing equal signature positions."""
        if len(self.signature) != len(other.signature):
            raise ValueError("Signatures must have identical lengths for Jaccard estimation.")

        matches = sum(1 for a, b in zip(self.signature, other.signature) if a == b)
        return matches / float(len(self.signature))


class MinHashGenerator:
    """Generates MinHash signatures using universal hashing permutations."""

    def __init__(self, num_perm: int = 128, seed: int = 42):
        if num_perm < 1:
            raise ValueError(f"num_perm must be >= 1, received {num_perm}")
        self.num_perm = num_perm
        self.seed = seed
        self.hash_coeffs: List[Tuple[int, int]] = self._generate_hash_coefficients()

    def _generate_hash_coefficients(self) -> List[Tuple[int, int]]:
        """Generates deterministic pseudo-random (a, b) pairs for universal hashing."""
        rng = random.Random(self.seed)
        coeffs: List[Tuple[int, int]] = []
        for _ in range(self.num_perm):
            a = rng.randint(1, _MERSENNE_PRIME_61 - 1)
            b = rng.randint(0, _MERSENNE_PRIME_61 - 1)
            coeffs.append((a, b))
        return coeffs

    def compute_signature(self, doc_id: str, shingle_hashes: Set[int]) -> MinHashSignature:
        """
        Computes the MinHash signature array across all permutation functions.
        If shingle set is empty, returns all zeros.
        """
        if not shingle_hashes:
            return MinHashSignature(doc_id=doc_id, signature=tuple([0] * self.num_perm))

        sig_values: List[int] = []

        for a, b in self.hash_coeffs:
            min_val = _MERSENNE_PRIME_61
            for sh_hash in shingle_hashes:
                # Universal hash modulo Mersenne prime
                val = ((a * sh_hash + b) & _MASK64) % _MERSENNE_PRIME_61
                if val < min_val:
                    min_val = val
            sig_values.append(min_val)

        return MinHashSignature(doc_id=doc_id, signature=tuple(sig_values))


def exact_jaccard(set_a: Set[int], set_b: Set[int]) -> float:
    """Computes exact Jaccard similarity: |A ∩ B| / |A ∪ B|."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / float(union)
