"""
CorpusDedup: Vectorized MinHash LSH Deduplication Engine.
Production data engineering suite for LLM corpora.
"""

from dedup_core.shingling import Shingler, ShingleType
from dedup_core.minhash import MinHashGenerator, MinHashSignature
from dedup_core.lsh import LSHIndex
from dedup_core.union_find import DisjointSet
from dedup_core.pipeline import DeduplicationPipeline, DeduplicationResult, DuplicateCluster

__version__ = "0.1.0"
__author__ = "Kaylor Tate Cespedes"

__all__ = [
    "Shingler",
    "ShingleType",
    "MinHashGenerator",
    "MinHashSignature",
    "LSHIndex",
    "DisjointSet",
    "DeduplicationPipeline",
    "DeduplicationResult",
    "DuplicateCluster",
]
