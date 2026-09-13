"""
Locality-Sensitive Hashing (LSH) index using band partitioning.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple
from collections import defaultdict
from dedup_core.minhash import MinHashSignature


class LSHIndex:
    """Partitions MinHash signatures into bands to index near-duplicate candidate pairs."""

    def __init__(self, num_perm: int = 128, num_bands: int = 16):
        if num_perm % num_bands != 0:
            raise ValueError(
                f"num_perm ({num_perm}) must be divisible by num_bands ({num_bands}). "
                f"Remainder is {num_perm % num_bands}."
            )

        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands

        # Table: (band_idx, band_tuple) -> list of doc_ids
        self.buckets: Dict[Tuple[int, Tuple[int, ...]], List[str]] = defaultdict(list)
        self.indexed_docs: Set[str] = set()

    def index(self, sig: MinHashSignature) -> None:
        """Indexes a single document signature into LSH buckets."""
        if sig.doc_id in self.indexed_docs:
            return

        self.indexed_docs.add(sig.doc_id)

        for b in range(self.num_bands):
            start_idx = b * self.rows_per_band
            end_idx = start_idx + self.rows_per_band
            band_chunk = sig.signature[start_idx:end_idx]
            bucket_key = (b, band_chunk)
            self.buckets[bucket_key].append(sig.doc_id)

    def get_candidate_pairs(self) -> Set[Tuple[str, str]]:
        """
        Extracts all unique candidate document pairs that collided in at least one bucket.
        Guarantees lexicographical ordering in pairs: (id_a < id_b).
        """
        candidate_pairs: Set[Tuple[str, str]] = set()

        for doc_list in self.buckets.values():
            if len(doc_list) > 1:
                # Add all combinations
                for i in range(len(doc_list)):
                    for j in range(i + 1, len(doc_list)):
                        doc_a, doc_b = doc_list[i], doc_list[j]
                        if doc_a != doc_b:
                            pair = (doc_a, doc_b) if doc_a < doc_b else (doc_b, doc_a)
                            candidate_pairs.add(pair)

        return candidate_pairs

    def query(self, sig: MinHashSignature) -> Set[str]:
        """Finds all indexed document IDs that share at least one band with sig."""
        candidates: Set[str] = set()
        for b in range(self.num_bands):
            start_idx = b * self.rows_per_band
            end_idx = start_idx + self.rows_per_band
            band_chunk = sig.signature[start_idx:end_idx]
            bucket_key = (b, band_chunk)
            if bucket_key in self.buckets:
                candidates.update(self.buckets[bucket_key])

        candidates.discard(sig.doc_id)
        return candidates
