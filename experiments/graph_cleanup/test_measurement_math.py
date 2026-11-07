import unittest

from analyze_gpu import spearman
from judge_revision import normalize


class MeasurementMathTest(unittest.TestCase):
    def test_citation_cleanup_preserves_factual_parentheses(self):
        self.assertEqual(normalize('Read core files (m1:u0-m16:u3).'),'Read core files.')
        self.assertEqual(normalize('Read core files (m1:u0, m13:u0).'),'Read core files.')
        self.assertEqual(normalize('Use master (not main); reviewed m1:u0.'),'Use master (not main); reviewed m1:u0.')
        self.assertEqual(normalize('The evidence (m1:u0 failed) matters.'),'The evidence (m1:u0 failed) matters.')

    def test_excluding_historical_pages_preserves_causal_keys_at_original_queries(self):
        # FlashInfer aligns Q to the end of the compact retained K sequence.
        # Whole-batch alignment is correct only for used rows after all removed
        # pages. Exercise a partial last page and removals inside the Q chunk.
        for length,count,boundary,removed in [
            (117,53,100,{0,2}), (209,64,192,{0,4,5}),
            (128,64,96,{0}), (257,1,256,{1,5})]:
            retained=[p for p in range(length) if p//32 not in removed]
            for row in range(count):
                absolute=length-count+row
                if absolute<boundary:
                    continue
                compact_limit=len(retained)-count+row
                actual=retained[:compact_limit+1]
                expected=[p for p in retained if p<=absolute]
                self.assertEqual(actual,expected)

    def test_current_query_removal_invalidates_alignment(self):
        length,count=128,32
        retained=[p for p in range(length) if p//32!=3]
        compact_limit=len(retained)-count
        self.assertNotEqual(retained[:compact_limit+1],[p for p in retained if p<=96])

    def test_spearman_handles_ties_and_undefined_constant(self):
        self.assertEqual(spearman([1,1,3,4],[2,2,6,8]),1.)
        self.assertEqual(spearman([1,1,3,4],[8,8,6,2]),-1.)
        self.assertIsNone(spearman([1,1,1],[1,2,3]))


if __name__=='__main__':
    unittest.main()
