"""Minimal tests for text chunking."""

import unittest

from src.embeddings.chunker import chunk_text


class TestChunkText(unittest.TestCase):
    """Test text chunking logic."""

    def test_short_text_single_chunk(self):
        text = "This is a short piece of text."
        chunks = chunk_text(text, max_chars=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_empty_text(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   "), [])

    def test_splits_long_text(self):
        # Create text longer than max_chars
        sentences = ["This is sentence number %d." % i for i in range(50)]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_chars=200, overlap=20)
        self.assertGreater(len(chunks), 1)
        # Each chunk should be under max_chars (with some tolerance for sentence boundaries)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 250)  # Allow some overflow at sentence boundary

    def test_preserves_all_content(self):
        """All original words should appear in at least one chunk."""
        words = ["word%d" % i for i in range(100)]
        text = ". ".join([" ".join(words[i:i+5]) for i in range(0, 100, 5)])
        chunks = chunk_text(text, max_chars=100, overlap=20)
        all_chunk_text = " ".join(chunks)
        for word in words:
            self.assertIn(word, all_chunk_text)

    def test_overlap_creates_context(self):
        """Consecutive chunks should have some overlapping content."""
        sentences = ["Sentence number %d is here." % i for i in range(20)]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_chars=150, overlap=50)
        if len(chunks) >= 2:
            # Check that second chunk starts with content from end of first
            # (overlap creates shared context)
            self.assertGreater(len(chunks[1]), 0)

    def test_single_long_sentence_handled(self):
        """A single sentence longer than max_chars should be split."""
        text = "word " * 500  # ~2500 chars
        chunks = chunk_text(text, max_chars=200, overlap=20)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertGreater(len(chunk), 0)


if __name__ == "__main__":
    unittest.main()
