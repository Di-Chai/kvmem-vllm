import unittest
from protocol import graph, history, rankings, token_mapping, observed_failure, scaffold_overlap


def unit(uid, text, step, kind='thinking', **kw):
    return dict(unit_id=uid, text=text, step=step, content_type=kind, entities=[], **kw)


class ProtocolTest(unittest.TestCase):
    def test_code_error_names_are_not_execution_failures(self):
        self.assertFalse(observed_failure(unit('a','25\t raise ValueError("error")',1,'tool_result')))
        self.assertFalse(observed_failure(unit('a','export { SomeClass };',1,'tool_result',is_error=True)))
        self.assertTrue(observed_failure(unit('b','Exit code 1\nfailed test',1,'tool_result')))
        self.assertFalse(observed_failure(unit('c','Exit code 1\nfailed test',1,'thinking')))

    def test_scaffolding_is_excluded_across_paragraph_boundaries(self):
        field = 'real data\n<system-reminder>first\n\nsecond</system-reminder>\ntail'
        self.assertFalse(scaffold_overlap(field,0,9))
        start = field.index('second')
        self.assertTrue(scaffold_overlap(field,start,start+6))

    def test_source_body_exact_with_unicode_and_delimiters(self):
        units = [unit('a','中文\n<unit>quoted</unit>\n',0), unit('b','second',1)]
        text, spans = history(units)
        for u in units:
            a,b = spans[u['unit_id']]
            self.assertEqual(text[a:b], u['text'])

    def test_future_duplicate_cannot_change_frozen_prefix(self):
        text = 'Preserve original byte offsets and verify all critical facts. '*3
        past = [unit('a',text,0), unit('b','new unrelated observation',1)]
        self.assertFalse(any(e['relation']=='redundant_with' for e in graph(past)))
        later = graph(past + [unit('c',text,2)])
        self.assertTrue(any(e['source']=='c' and e['target']=='a' for e in later))

    def test_shared_blocks_are_not_silently_evicted(self):
        offsets = [(i,i+1) for i in range(12)]
        m = token_mapping(offsets, {'a':[1,6], 'b':[6,11]}, block_size=4)
        self.assertEqual(m['a']['logical_blocks'], [0,1])
        self.assertEqual(m['a']['fully_contained_logical_blocks'], [])
        self.assertEqual(m['a']['collateral_unit_ids'], ['b'])
        self.assertEqual(token_mapping(offsets,{'x':[4,8]},4)['x']['fully_contained_logical_blocks'],[1])

    def test_tool_links_use_actual_ids_and_constraints_remain_protected(self):
        us = [unit('a','must preserve public API',0,'task_instruction'),
              unit('b','read file',1,'tool_call',call_id='one'),
              unit('c','file contents',2,'tool_result',call_id='two'),
              unit('d','file contents',3,'tool_result',call_id='one')]
        es = graph(us)
        self.assertEqual([(e['source'],e['target']) for e in es if e['relation']=='tool_result_of'], [('d','b')])
        scores = {s['unit_id']:s for s in rankings(us,es)}
        self.assertLess(scores['a']['graph_propagation'],scores['c']['graph_propagation'])


if __name__=='__main__': unittest.main()
