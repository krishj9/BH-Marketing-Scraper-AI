"""Minimal tests for exact matching and vector similarity."""

import unittest

from src.matching.exact import ExactMatcher
from src.matching.vector import compute_similarities, cosine_similarity, filter_by_threshold


class TestExactMatcher(unittest.TestCase):
    """Test exact keyword substring matching."""

    def setUp(self):
        self.keywords = [
            "corporate daycare",
            "childcare benefit",
            "on site childcare",
            "dependent care",
        ]
        self.matcher = ExactMatcher(self.keywords)

    def test_exact_match_found(self):
        text = "We offer a Corporate Daycare program for all employees."
        result = self.matcher.match(text)
        self.assertIsNotNone(result)
        self.assertTrue(result.matched)
        self.assertEqual(result.keyword, "corporate daycare")

    def test_no_match(self):
        text = "Our company provides health insurance and 401k plans."
        result = self.matcher.match(text)
        self.assertIsNone(result)

    def test_case_insensitive(self):
        text = "CHILDCARE BENEFIT available for full-time employees."
        result = self.matcher.match(text)
        self.assertIsNotNone(result)
        self.assertEqual(result.keyword, "childcare benefit")

    def test_empty_text(self):
        result = self.matcher.match("")
        self.assertIsNone(result)

    def test_match_all(self):
        text = "We provide corporate daycare and dependent care benefits."
        results = self.matcher.match_all(text)
        keywords_found = {r.keyword for r in results}
        self.assertIn("corporate daycare", keywords_found)
        self.assertIn("dependent care", keywords_found)

    def test_partial_word_match(self):
        """Substring matching means 'childcare benefit' matches inside longer text."""
        text = "The childcare benefit program is available to parents."
        result = self.matcher.match(text)
        self.assertIsNotNone(result)

    def test_no_keywords(self):
        matcher = ExactMatcher([])
        result = matcher.match("some text")
        self.assertIsNone(result)


class TestCosineSimilarity(unittest.TestCase):
    """Test cosine similarity computation."""

    def test_identical_vectors(self):
        vec = [1.0, 0.0, 0.5]
        sim = cosine_similarity(vec, vec)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_orthogonal_vectors(self):
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]
        sim = cosine_similarity(vec_a, vec_b)
        self.assertAlmostEqual(sim, 0.0, places=5)

    def test_opposite_vectors(self):
        vec_a = [1.0, 0.0]
        vec_b = [-1.0, 0.0]
        sim = cosine_similarity(vec_a, vec_b)
        self.assertAlmostEqual(sim, -1.0, places=5)

    def test_zero_vector(self):
        vec_a = [0.0, 0.0]
        vec_b = [1.0, 1.0]
        sim = cosine_similarity(vec_a, vec_b)
        self.assertEqual(sim, 0.0)


class TestComputeSimilarities(unittest.TestCase):
    """Test batch similarity computation."""

    def test_batch_similarities(self):
        chunks = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
        keyword = [1.0, 0.0]
        sims = compute_similarities(chunks, keyword)
        self.assertEqual(len(sims), 3)
        self.assertAlmostEqual(sims[0], 1.0, places=4)
        self.assertAlmostEqual(sims[1], 0.0, places=4)
        self.assertGreater(sims[2], 0.5)

    def test_empty_chunks(self):
        sims = compute_similarities([], [1.0, 0.0])
        self.assertEqual(sims, [])


class TestFilterByThreshold(unittest.TestCase):
    """Test threshold filtering."""

    def test_filter(self):
        sims = [0.9, 0.5, 0.8, 0.3, 0.75]
        results = filter_by_threshold(sims, 0.70)
        indices = [idx for idx, _ in results]
        self.assertIn(0, indices)  # 0.9
        self.assertIn(2, indices)  # 0.8
        self.assertIn(4, indices)  # 0.75
        self.assertNotIn(1, indices)  # 0.5
        self.assertNotIn(3, indices)  # 0.3

    def test_sorted_descending(self):
        sims = [0.75, 0.9, 0.8]
        results = filter_by_threshold(sims, 0.70)
        scores = [s for _, s in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
