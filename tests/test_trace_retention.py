import unittest

from scripts.analyze_trace_retention import overlap


class RetentionOverlapTest(unittest.TestCase):
    def test_disjoint_selected_pages_and_partial_semantic_block(self):
        self.assertEqual(overlap(16, 80, [(0, 32), (64, 32)]), 32)

    def test_touching_half_open_endpoints_do_not_count(self):
        self.assertEqual(overlap(32, 64, [(0, 32), (64, 32)]), 0)

    def test_retention_is_bounded_by_content_block_size(self):
        self.assertEqual(overlap(10, 20, [(0, 32)]), 10)


if __name__ == '__main__':
    unittest.main()
