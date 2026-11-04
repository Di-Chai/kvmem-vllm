"""Tests protect methodological boundaries, not incidental implementation."""
from pathlib import Path
import json
import tempfile
import unittest

from core import (json_string_fields, decoded_boundaries, structural_spans,
                  detect_reuse, entities, normalize_trace, verify_unit)
from events import safe_test_result, observed_events, supersession_relations
from analysis_stats import km_curve, future_stats, rank_correlation, cross_seed
from prediction import task_folds, prediction_examples, average_precision


def unit(uid, text, turn, kind='thinking', **kw):
    row=dict(unit_id=uid,message_id=uid.split(':')[0],trajectory_id='deepswe/seed-1001/test',
        family='deepswe',task_id='test',seed=1001,attempt_source='preserved_original',
        message_index=turn*2,turn_id=turn,step=turn,role='assistant',retry=False,
        trace_path='trace.jsonl',trace_line=turn+1,raw_source={},timestamp=None,
        content_type=kind,subtype='prose_or_command',field_pointer='/original/content',
        character_span=[0,len(text)],text=text,entities=entities(text),main_eligible=True)
    row.update(kw)
    return row


class Coordinates(unittest.TestCase):
    def test_json_escaped_unicode_offsets(self):
        value='中文\nquote " and backslash \\ emoji 🧬'
        text='```json\n'+json.dumps({'analysis':value,'commands':[{'keystrokes':'cat /a.py\n'}]},ensure_ascii=True)+'\n```'
        fields=json_string_fields(text)
        decoded,a,b=fields['/analysis']
        self.assertEqual(decoded,value)
        bounds=decoded_boundaries(text[a:b])
        self.assertEqual(len(bounds),len(value)+1)
        for start,end in [(0,2),(2,6),(len(value)-1,len(value))]:
            self.assertEqual(json.loads('"'+text[a+bounds[start]:a+bounds[end]]+'"'),value[start:end])

    def test_split_preserves_exact_text_and_bounds(self):
        text=('header\n\n```python\n'+('def function(): return 123\n'*150)+'```\n')
        spans=structural_spans(text)
        self.assertEqual(''.join(text[a:b] for a,b in spans),text)
        self.assertTrue(all(b-a<=1400 for a,b in spans))

    def test_invalid_json_has_no_invented_fields(self):
        self.assertEqual(json_string_fields('{"analysis": "hello", bad}'),{})


class ReuseBoundaries(unittest.TestCase):
    def test_keyword_or_path_alone_is_not_reuse(self):
        a=unit('a','Read /app/module.py for possible options.',1,'tool_result',role='user')
        b=unit('b','Now modify /app/module.py.',5)
        self.assertEqual(detect_reuse([a,b]),[])

    def test_same_turn_not_future_and_tool_echo_not_agent_reuse(self):
        text='The persistent capacity invariant requires allocator storage to remain aligned after resizing.'
        a=unit('a',text,1)
        b=unit('b',text,1)
        c=unit('c',text,3,'tool_result',role='user')
        self.assertEqual(detect_reuse([a,b,c]),[])

    def test_retry_excluded_main_but_visible_in_sensitivity(self):
        text='The persistent capacity invariant requires allocator storage to remain aligned after resizing.'
        a=unit('a',text,1)
        b=unit('b',text,3,retry=True,main_eligible=False)
        self.assertEqual(detect_reuse([a,b]),[])
        self.assertTrue(any(e['level']=='explicit' for e in detect_reuse([a,b],True)))

    def test_latest_attribution_tracks_ambiguity(self):
        text='The persistent capacity invariant requires allocator storage to remain aligned after resizing.'
        rows=[unit(str(i),text,i) for i in (1,2,7)]
        es=detect_reuse(rows)
        last=[e for e in es if e['target_turn']==7 and e['level']=='explicit']
        self.assertEqual({e['source_unit_id'] for e in last},{'2'})
        self.assertTrue(all(e['competing_prior_occurrences']>0 for e in last))


class EventsAndCensoring(unittest.TestCase):
    def test_repeated_edit_receipt_does_not_supersede_old_change(self):
        text='The file /app/specific_file.py has been updated successfully.'
        a=unit('a',text,1,'tool_result',role='user',tool='Edit')
        b=unit('b',text,2,'tool_result',role='user',tool='Edit')
        events=[dict(event_id=f'e{i}',event_type='code_modification',step=i,turn_id=i,
            scope='/app/specific_file.py',scope_entities=[],target_unit_ids=[u['unit_id']],
            source_call_unit_id=None,target_message_id=u['message_id'],tool='Edit',
            evidence=[],_output=text) for i,u in [(1,a),(2,b)]]
        relations=supersession_relations([a,b],events,[],3)
        self.assertTrue(relations)
        self.assertTrue(all(r['relation']=='uncertain' for r in relations))

    def test_exit_zero_and_claim_not_test_success(self):
        self.assertIsNone(safe_test_result('pytest','Everything fixed.',0))
        self.assertIsNone(safe_test_result('echo passed','10 passed',0))
        self.assertIsNone(safe_test_result('echo "pytest"; echo "10 passed"','10 passed',0))
        self.assertIsNone(safe_test_result('cat pytest.log','10 passed',0))
        self.assertIsNone(safe_test_result('python -c "print(\"pytest 10 passed\")"','10 passed',0))
        self.assertEqual(safe_test_result('pytest','10 passed in 0.3s',0),'test_success')
        self.assertEqual(safe_test_result('cd /app && python3 -m pytest','10 passed in 0.3s',0),'test_success')
        self.assertEqual(safe_test_result('pytest','10 passed, 1 failed',0),'test_failure')

    def test_completion_claim_not_verified_completion(self):
        item={'family':'terminalbench','task_id':'test','seed':None,'trajectory_id':'tb/test'}
        u=unit('a','Task complete, tests passed.',1)
        events,_=observed_events(item,[u],[],[],{})
        self.assertEqual(events,[])

    def test_km_events_before_same_time_censor(self):
        rows=km_curve([(1,True),(1,False),(2,True),(3,False)])
        self.assertEqual(rows[0]['at_risk'],4)
        self.assertAlmostEqual(rows[0]['cumulative_reuse'],.25)
        self.assertAlmostEqual(rows[1]['cumulative_reuse'],.625)

    def test_semantic_use_interrupts_explicit_dormancy(self):
        a=unit('a','Historical information',0)
        item={'saved_assistant_turns':30,'main_steps':30,'final_saved_response_available':True}
        edges=[dict(source_unit_id='a',level='semantic',target_turn=18,target_step=18),
               dict(source_unit_id='a',level='explicit',target_turn=20,target_step=20)]
        _,_,_,gaps,_=future_stats([a],edges,{a['trajectory_id']:item})
        row=next(r for r in gaps if r['level']=='explicit')
        self.assertEqual(row['gap_since_last_observed_use'],2)

    def test_full_window_denominator_excludes_censored_early_positive(self):
        a=unit('a','Early information',0);b=unit('b','Late information',9)
        item={'saved_assistant_turns':10,'main_steps':10,'final_saved_response_available':True}
        edges=[dict(source_unit_id='b',level='explicit',target_turn=10,target_step=10)]
        _,summary,_,_,_=future_stats([a,b],edges,{a['trajectory_id']:item})
        row=next(r for r in summary if r['content_type']=='ALL' and r['level']=='explicit' and r['window']==4)
        self.assertEqual(row['full_window_units'],1)
        self.assertEqual(row['full_window_hits'],0)
        self.assertEqual(row['excluded_incomplete_window'],1)


class PredictionBoundaries(unittest.TestCase):
    def test_uncertain_supersession_and_missing_seed_are_not_negative(self):
        us=[unit(f'u{s}','Fix /app/specific_file.py with constraints.',0,seed=s) for s in (1001,1002,1003)]
        relations=[dict(source_unit_id=u['unit_id'],event_type='code_modification',relation='uncertain') for u in us]
        index=[dict(family='deepswe',task_id='test',seed=s) for s in (1001,1002,1003,1004)]
        aligned,tasks,events=cross_seed(us,[],relations,index)
        self.assertTrue(aligned)
        self.assertTrue(all(not r['four_seed_eligible'] for r in aligned))
        self.assertTrue(all(r['supersession_agreement'] is None for r in aligned))
        self.assertTrue(all(r['comparable_seed_pairs']==0 for r in events))
        self.assertIsNone(tasks[0]['stable_demand_ratio'])

    def test_all_seeds_share_task_fold(self):
        rows=[{'family':'deepswe','task_id':t,'seed':s} for t in ('a','b','c') for s in range(4)]
        mapping=task_folds(rows)
        self.assertEqual(len(mapping),3)

    def test_future_edges_do_not_change_features(self):
        u=unit('a','Historical buffer constraint',0)
        q=unit('q','Current observation buffer',1,'tool_result',role='user')
        item={'trajectory_id':u['trajectory_id'],'main_steps':10}
        fold={('deepswe','test'):0}
        future=[dict(source_unit_id='a',level='explicit',target_step=3)]
        a=prediction_examples(item,[u,q],[],[],fold)
        b=prediction_examples(item,[u,q],future,[],fold)
        aa=next(e for e in a if e['unit_id']=='a' and e['checkpoint_step']==1 and e['window']==4)
        bb=next(e for e in b if e['unit_id']=='a' and e['checkpoint_step']==1 and e['window']==4)
        self.assertEqual(aa['scores'],bb['scores'])
        self.assertNotEqual(aa['labels']['explicit'],bb['labels']['explicit'])
        self.assertIsNone(bb['labels']['high_confidence_dependency'])

    def test_tied_ap_uses_grouped_threshold(self):
        self.assertAlmostEqual(average_precision([1,0],[.5,.5]),.5)
        self.assertIsNone(average_precision([0,0],[.5,.2]))
        self.assertIsNone(rank_correlation([1,1,1],[0,1,0]))


if __name__=='__main__':
    unittest.main()
