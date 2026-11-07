"""Causal replay, evidence-coordinate and stage-identity invariants."""
import copy
import unittest

from observer import Memory, extract_step, request_for, resolve


def step(turn=1):
    records = [(4,dict(turn=turn,message_index=1,original=dict(role='assistant',content=[
        dict(type='thinking',thinking='Implement the parser; success of an edit is not verification.'),
        dict(type='tool_use',name='Bash',input=dict(command='do not execute this',description='Edit parser'))]))),
        (5,dict(turn=turn,message_index=2,original=dict(role='user',content=[
            dict(type='tool_result',content='successful edit\n'+'x'*1000+'\nTEST FAILED')])))]
    item=dict(trace_path='trace.jsonl',trajectory_id='t',task_id='test',seed=1001)
    result=extract_step(item,turn,records)
    result.update(window_id='w',task_instruction='Fix the parser')
    return result,records


def response(link='new_stage',outcome='in_progress'):
    return dict(answers={k:dict(choice=v) for k,v in dict(stage_link=link,
        outcome=outcome,activity='implementation',evidence='e1').items()})


class Invariants(unittest.TestCase):
    def test_exact_citation_coordinates_include_result_tail(self):
        current,records=step()
        by_line=dict(records)
        for candidate in current['evidence'].values():
            src=candidate['source'];raw=resolve(by_line[src['trace_line']],src['pointer'])
            begin,end=src['character_span']
            self.assertEqual(raw[begin:end],candidate['quote'])
        self.assertTrue(any('TEST FAILED' in c['quote'] for c in current['evidence'].values()))

    def test_future_step_rejected(self):
        current,records=step()
        with self.assertRaises(ValueError):
            extract_step(dict(trace_path='t',trajectory_id='t',task_id='t',seed=1),2,records)

    def test_memory_snapshot_isolated_and_future_excluded(self):
        current,_=step();memory=Memory()
        payload=request_for(current,memory,'m')
        memory.apply(current,response())
        self.assertEqual(payload['state']['execution_memory']['stages'],[])
        with self.assertRaises(ValueError):request_for(current,memory,'m')

    def test_no_completion_inferred_from_switch(self):
        current,_=step();memory=Memory();memory.apply(current,response())
        later,_=step(2);memory.apply(later,response())
        self.assertEqual(memory.stages[0]['status'],'in_progress')
        self.assertEqual(len(memory.stages),2)

    def test_continuation_and_later_failure_revise_same_stage(self):
        current,_=step();memory=Memory();memory.apply(current,response(outcome='completed'))
        later,_=step(2);result=memory.apply(later,response('S001','failed'))
        self.assertEqual(len(memory.stages),1)
        self.assertFalse(result['stage_boundary'])
        self.assertEqual(result['status_revision'],dict(previous='completed',current='failed'))

    def test_duplicate_or_reordered_step_rejected(self):
        current,_=step();memory=Memory();memory.apply(current,response())
        with self.assertRaises(ValueError):memory.apply(current,response())

    def test_uncertain_assignment_does_not_invent_a_later_boundary(self):
        current,_=step();memory=Memory();memory.apply(current,response())
        middle,_=step(2);memory.apply(middle,response('uncertain'))
        later,_=step(3);result=memory.apply(later,response('S001'))
        self.assertIsNone(result['stage_boundary'])
        self.assertEqual(len(memory.stages),1)

    def test_current_only_has_no_memory(self):
        current,_=step();memory=Memory();memory.apply(current,response())
        later,_=step(2);payload=request_for(later,memory,'m','current')
        self.assertEqual(payload['state']['execution_memory'],dict(stages=[],recent_steps=[]))

    def test_extracting_current_step_is_invariant_to_future_records(self):
        current,records=step()
        records_copy=copy.deepcopy(records)
        records_copy.append((6,dict(turn=2,message_index=3,original=dict(role='assistant',content='future'))))
        filtered=[(i,r) for i,r in records_copy if r['turn']==1]
        self.assertEqual(current['current_step'],extract_step(dict(trace_path='trace.jsonl',
            trajectory_id='t',task_id='test',seed=1001),1,filtered)['current_step'])


if __name__=='__main__':unittest.main()
