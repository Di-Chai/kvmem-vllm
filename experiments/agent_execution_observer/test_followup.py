"""Intervention isolation and immutable input checks for new model experiments."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from followup import LETTER_SUFFIX, bounded_goal, check_inputs, explicit_letters
from observer import digest
from followup_audit import same_payload
from goal_hint_probe import replace_hint


def payload():
    return dict(model='observer',instructions='unchanged system policy',
        state=dict(current_step_id=5,current_step='visible evidence',execution_memory=dict(stages=[])),
        questions={q:dict(type='choice',instructions=q+' question',criteria={'one':'First','two':'Second'})
                   for q in ('activity','stage_link','outcome','evidence')})


class FollowupInvariants(unittest.TestCase):
    def test_letter_intervention_changes_only_each_question_instruction(self):
        original=payload();snapshot=copy.deepcopy(original);changed=explicit_letters(original)
        for q in changed['questions']:
            self.assertEqual(changed['questions'][q]['instructions'],original['questions'][q]['instructions']+LETTER_SUFFIX)
            changed['questions'][q]['instructions']=original['questions'][q]['instructions']
            self.assertEqual(list(changed['questions'][q]['criteria']),['one','two'])
        self.assertEqual(changed,original);self.assertEqual(original,snapshot)

    def test_bound_goal_visible_to_all_independent_reads(self):
        original=payload();changed=bounded_goal(original,'test-attempt','All tests in this attempt must pass')
        self.assertEqual(set(changed['questions']),{'outcome','evidence'})
        self.assertEqual(changed['state']['assessment_target']['target_id'],'test-attempt')
        self.assertEqual(changed['state']['current_step_id'],5)
        self.assertNotIn('assessment_target',original['state'])
        self.assertNotIn('accepted_outcomes',changed['state'])

    def test_goal_binding_preserves_observed_context_and_option_order(self):
        original=payload();changed=bounded_goal(original,'file','Create source file')
        for key,value in original['state'].items():self.assertEqual(changed['state'][key],value)
        for q in changed['questions']:
            self.assertEqual(list(changed['questions'][q]['criteria'].items()),list(original['questions'][q]['criteria'].items()))

    def test_reject_tampered_frozen_input(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'request.json';p.write_text('{}')
            (root/'manifest.json').write_text(json.dumps(dict(input_files={'request.json':digest(p)})))
            check_inputs(root);p.write_text('{"future":true}')
            with self.assertRaisesRegex(ValueError,'Frozen input changed'):check_inputs(root)

    def test_audit_rejects_option_order_change(self):
        original=payload();changed=copy.deepcopy(original)
        changed['questions']['activity']['criteria']={'two':'Second','one':'First'}
        with self.assertRaisesRegex(ValueError,'option order'):same_payload(changed,original)

    def test_audit_rejects_goal_reference_leakage(self):
        original=bounded_goal(payload(),'test','The tests pass');changed=copy.deepcopy(original)
        changed['state']['expected_outcome']='failed'
        with self.assertRaisesRegex(ValueError,'Model state'):same_payload(changed,original)

    def test_goal_hint_replaced_in_both_state_and_options_without_reordering(self):
        original=payload();stage=dict(stage_id='S004',goal_hint=dict(text='branch text'))
        original['state']['execution_memory']['stages']=[stage]
        original['questions']['stage_link']['criteria']={'S004':json.dumps(stage),'new_stage':'New'}
        hint=dict(text='Create options parser',source=dict(step=3,pointer='/subject'))
        changed=replace_hint(original,hint)
        self.assertEqual(changed['state']['execution_memory']['stages'][0]['goal_hint'],hint)
        self.assertEqual(json.loads(changed['questions']['stage_link']['criteria']['S004'])['goal_hint'],hint)
        self.assertEqual(list(changed['questions']['stage_link']['criteria']),['S004','new_stage'])
        self.assertEqual(original['state']['execution_memory']['stages'][0]['goal_hint']['text'],'branch text')

    def test_goal_hint_cannot_use_future_declared_task(self):
        with self.assertRaisesRegex(ValueError,'not from the past'):
            replace_hint(payload(),dict(text='Future task',source=dict(step=7)))


if __name__=='__main__':unittest.main()
