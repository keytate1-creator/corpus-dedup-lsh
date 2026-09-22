"""
Unit tests for k-shingle generation and hashing using unittest.
"""

import unittest
from dedup_core.shingling import Shingler, ShingleType, fnv1a_64


class TestShingling(unittest.TestCase):

    def test_fnv1a_64_deterministic(self):
        h1 = fnv1a_64("hello world")
        h2 = fnv1a_64("hello world")
        h3 = fnv1a_64("hello world!")
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)
        self.assertIsInstance(h1, int)

    def test_word_shingler(self):
        text = "The quick brown fox jumps over the lazy dog."
        shingler = Shingler(k=3, shingle_type=ShingleType.WORD)
        hashes = shingler.get_shingle_hashes(text)

        # Cleaned words: the, quick, brown, fox, jumps, over, the, lazy, dog (9 words)
        # k=3 shingles: 9 - 3 + 1 = 7 shingles
        self.assertEqual(len(hashes), 7)

    def test_char_shingler(self):
        text = "abcde"
        shingler = Shingler(k=3, shingle_type=ShingleType.CHAR)
        hashes = shingler.get_shingle_hashes(text)

        # 3-char shingles: 'abc', 'bcd', 'cde' = 3 shingles
        self.assertEqual(len(hashes), 3)

    def test_short_document(self):
        text = "hi"
        shingler = Shingler(k=5, shingle_type=ShingleType.WORD)
        hashes = shingler.get_shingle_hashes(text)
        self.assertEqual(len(hashes), 1)


if __name__ == "__main__":
    unittest.main()
