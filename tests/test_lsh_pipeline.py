"""
Integration tests for LSH indexing and the deduplication pipeline using unittest.
"""

import unittest
from dedup_core.pipeline import DeduplicationPipeline
from dedup_core.shingling import ShingleType


class TestLSHPipeline(unittest.TestCase):

    def test_pipeline_deduplication(self):
        # 128 perms / 32 bands = 4 rows per band.
        # S-curve threshold t ~= (1/32)^(1/4) ~= 0.42. High recall for similarity >= 0.60.
        pipeline = DeduplicationPipeline(
            num_perm=128,
            num_bands=32,
            similarity_threshold=0.60,
            shingle_size=2,
            shingle_type=ShingleType.WORD,
        )

        documents = [
            {
                "id": "doc1",
                "text": "The quick brown fox jumps over the lazy dog and rests peacefully under the large oak tree in the summer meadow.",
            },
            {
                "id": "doc2",
                "text": "The quick brown fox jumps over the lazy dog and sleeps peacefully under the large oak tree in the summer meadow.",
            },  # High-overlap duplicate of doc1
            {
                "id": "doc3",
                "text": "Quantum computers operate on quantum bits known as qubits which exhibit superposition and entanglement.",
            },
            {
                "id": "doc4",
                "text": "Modern quantum computers operate on quantum bits known as qubits which exhibit superposition and entanglement.",
            },  # High-overlap duplicate of doc3
            {
                "id": "doc5",
                "text": "An entirely unrelated culinary recipe describing how to ferment sourdough bread with organic whole wheat flour.",
            },  # Completely distinct
        ]

        result = pipeline.deduplicate(documents)

        self.assertEqual(result.total_documents, 5)
        self.assertEqual(result.duplicates_removed, 2)
        self.assertEqual(len(result.unique_documents), 3)

        retained_ids = {d["id"] for d in result.unique_documents}
        self.assertIn("doc1", retained_ids)
        self.assertIn("doc3", retained_ids)
        self.assertIn("doc5", retained_ids)
        self.assertNotIn("doc2", retained_ids)
        self.assertNotIn("doc4", retained_ids)

    def test_pipeline_empty(self):
        pipeline = DeduplicationPipeline()
        result = pipeline.deduplicate([])
        self.assertEqual(result.total_documents, 0)
        self.assertEqual(result.duplicates_removed, 0)
        self.assertEqual(len(result.unique_documents), 0)


if __name__ == "__main__":
    unittest.main()
