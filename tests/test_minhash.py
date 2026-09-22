"""
Unit tests for MinHash signature computation and Jaccard estimation using unittest.
"""

import unittest
from dedup_core.minhash import MinHashGenerator, exact_jaccard


class TestMinHash(unittest.TestCase):

    def test_exact_jaccard(self):
        set_a = {1, 2, 3, 4}
        set_b = {3, 4, 5, 6}
        # Intersection = {3, 4} (2), Union = {1, 2, 3, 4, 5, 6} (6) => 2/6 = 0.3333
        self.assertAlmostEqual(exact_jaccard(set_a, set_b), 2.0 / 6.0, places=6)

        # Identity
        self.assertEqual(exact_jaccard(set_a, set_a), 1.0)

        # Disjoint
        self.assertEqual(exact_jaccard({1, 2}, {3, 4}), 0.0)

    def test_minhash_generator_estimation(self):
        # 512 permutations yields standard error sigma = 1 / sqrt(512) ~= 0.044
        gen = MinHashGenerator(num_perm=512, seed=42)

        set_a = set(range(200))
        # 75% overlap
        set_b = set(range(50, 250))

        real_jaccard = exact_jaccard(set_a, set_b)
        # Intersection = 150, Union = 250 => 150/250 = 0.60

        sig_a = gen.compute_signature("doc_a", set_a)
        sig_b = gen.compute_signature("doc_b", set_b)

        estimated_jaccard = sig_a.estimate_jaccard(sig_b)

        # Statistical deviation within 2.5 * sigma (< 0.11)
        self.assertLess(abs(real_jaccard - estimated_jaccard), 0.11)


if __name__ == "__main__":
    unittest.main()
