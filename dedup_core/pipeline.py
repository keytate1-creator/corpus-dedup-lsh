"""
Orchestration pipeline for full corpus deduplication.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Sequence

from dedup_core.shingling import Shingler, ShingleType
from dedup_core.minhash import MinHashGenerator, MinHashSignature, exact_jaccard
from dedup_core.lsh import LSHIndex
from dedup_core.union_find import DisjointSet


@dataclass
class DuplicateCluster:
    """Represents a cluster of duplicate documents with one primary retained."""
    primary_id: str
    duplicate_ids: List[str]
    similarity_score: float = 1.0


@dataclass
class DeduplicationResult:
    """Consolidated metrics and filtered documents from a deduplication run."""
    total_documents: int
    duplicates_removed: int
    unique_documents: List[Dict[str, Any]]
    duplicate_clusters: List[DuplicateCluster] = field(default_factory=list)
    runtime_sec: float = 0.0

    @property
    def retention_rate(self) -> float:
        if self.total_documents == 0:
            return 1.0
        return len(self.unique_documents) / float(self.total_documents)


class DeduplicationPipeline:
    """End-to-end deduplication pipeline executing LSH candidate generation and cluster reduction."""

    def __init__(
        self,
        num_perm: int = 128,
        num_bands: int = 16,
        similarity_threshold: float = 0.75,
        shingle_size: int = 5,
        shingle_type: ShingleType = ShingleType.WORD,
        verify_exact_jaccard: bool = True,
        seed: int = 42,
    ):
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.similarity_threshold = similarity_threshold
        self.verify_exact_jaccard = verify_exact_jaccard

        self.shingler = Shingler(k=shingle_size, shingle_type=shingle_type)
        self.minhash_gen = MinHashGenerator(num_perm=num_perm, seed=seed)

    def deduplicate(
        self,
        documents: Sequence[Dict[str, Any]],
        text_key: str = "text",
        id_key: str = "id",
    ) -> DeduplicationResult:
        """
        Executes deduplication over a sequence of document dictionaries.
        Returns DeduplicationResult containing unique docs and cluster traces.
        """
        start_time = time.perf_counter()
        total_docs = len(documents)
        if total_docs == 0:
            return DeduplicationResult(
                total_documents=0,
                duplicates_removed=0,
                unique_documents=[],
                runtime_sec=0.0,
            )

        # 1. Shingle and MinHash all documents
        doc_store: Dict[str, Dict[str, Any]] = {}
        shingle_store: Dict[str, Set[int]] = {}
        signatures: Dict[str, MinHashSignature] = {}

        lsh = LSHIndex(num_perm=self.num_perm, num_bands=self.num_bands)

        for doc in documents:
            doc_id = str(doc[id_key])
            text = str(doc[text_key])
            doc_store[doc_id] = doc

            sh_hashes = self.shingler.get_shingle_hashes(text)
            shingle_store[doc_id] = sh_hashes

            sig = self.minhash_gen.compute_signature(doc_id, sh_hashes)
            signatures[doc_id] = sig
            lsh.index(sig)

        # 2. Extract Candidate Pairs from LSH Buckets
        candidates = lsh.get_candidate_pairs()

        # 3. Filter Candidates and Build Connected Components
        dset = DisjointSet[str]()
        for doc_id in doc_store:
            dset.make_set(doc_id)

        for id_a, id_b in candidates:
            if self.verify_exact_jaccard:
                sim = exact_jaccard(shingle_store[id_a], shingle_store[id_b])
            else:
                sim = signatures[id_a].estimate_jaccard(signatures[id_b])

            if sim >= self.similarity_threshold:
                dset.union(id_a, id_b)

        # 4. Formulate Clusters and Retain Primary (lexicographically first or insertion order)
        clusters = dset.get_clusters()
        retained_ids: Set[str] = set()
        duplicate_clusters: List[DuplicateCluster] = []

        for root, members in clusters.items():
            sorted_members = sorted(members)
            primary = sorted_members[0]
            retained_ids.add(primary)

            if len(sorted_members) > 1:
                duplicate_clusters.append(
                    DuplicateCluster(
                        primary_id=primary,
                        duplicate_ids=sorted_members[1:],
                        similarity_score=self.similarity_threshold,
                    )
                )

        unique_docs = [doc for doc in documents if str(doc[id_key]) in retained_ids]
        runtime_sec = time.perf_counter() - start_time
        duplicates_removed = total_docs - len(unique_docs)

        return DeduplicationResult(
            total_documents=total_docs,
            duplicates_removed=duplicates_removed,
            unique_documents=unique_docs,
            duplicate_clusters=duplicate_clusters,
            runtime_sec=runtime_sec,
        )
