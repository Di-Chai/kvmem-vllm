"""Structural origin must not be promoted to semantic activity or task success."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from structure_slicing import StructureAdapter, load_rules, partition, top_level_fields


BUNDLE = Path(__file__).resolve().parents[2]/'artifacts/agent-traces-handoff-20261102-r2'


class LexicalPartition(unittest.TestCase):
    def test_json_value_offsets_are_encoded_not_decoded(self):
        value = 'print("quoted {code}")\n中文'
        text = json.dumps(dict(analysis=value, plan='read', commands=[]), ensure_ascii=False)
        field = top_level_fields(text)[0]
        self.assertEqual(json.loads(text[field['begin']:field['end']]), value)
        self.assertNotEqual(text[field['begin']:field['end']], value)

    def test_nested_objects_do_not_create_top_level_fields(self):
        text='{"analysis":"x", "commands":[{"analysis":"nested", "keystrokes":"ls\\n"}], "plan":"y"}'
        self.assertEqual([f['key'] for f in top_level_fields(text)], ['analysis','commands','plan'])

    def test_duplicate_keys_preserve_lexemes_for_last_key_selection(self):
        fields=top_level_fields('{"analysis":"old","analysis":"new"}')
        self.assertEqual({f['key']:f for f in fields}['analysis']['value'],'new')

    def test_specific_status_overrides_observation_without_losing_parent(self):
        text='New Terminal Output:\nhello\n[Executor status]\nexit=0'
        begin=text.index('[Executor status]')
        rows=partition(text,[dict(kind='terminal_observation'),
            dict(kind='execution_status',character_span=[begin,len(text)])],'unclassified_input')
        self.assertEqual([r['kind'] for r in rows],['terminal_observation','execution_status'])
        self.assertEqual(rows[-1]['also_types'],['terminal_observation'])
        self.assertEqual(''.join(text[r['begin']:r['end']] for r in rows),text)

    def test_unmatched_input_is_not_forced_into_17_types(self):
        rows=partition('unknown',[dict(kind='user_text')],'unclassified_input')
        self.assertEqual(rows[0]['kind'],'unclassified_input')


@unittest.skipUnless((BUNDLE/'methodology/frozen/terminus_json_plain_parser.py').exists(),
                     'Optional frozen trace/parser artifact is unavailable')
class FrozenParserAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules,cls.parser,_=load_rules(BUNDLE)
        cls.item=dict(family='terminalbench',trajectory_id='terminalbench/test',trace_path='trace.jsonl')

    def record(self,content,reasoning=''):
        original=dict(role='assistant',content=content,reasoning_content=reasoning)
        _,parsed,_=self.rules.label_assistant(original,self.parser,following=None)
        return dict(record_type='message',turn=1,message_index=1,role='assistant',original=original,parsed=parsed)

    def extract(self,record):
        return StructureAdapter(self.rules,self.parser).record_blocks(self.item,record,1,'unused task')

    def test_code_in_analysis_stays_analysis(self):
        code='def parser():\n    return "TaskUpdate"'
        row=self.record(json.dumps(dict(analysis=code,plan='read',commands=[])))
        blocks=self.extract(row)
        analysis=next(b for b in blocks if b['structural_type']=='analysis')
        self.assertEqual(analysis['decoded_value'],code)
        self.assertFalse(any(b['structural_type']=='progress_operation' for b in blocks))

    def test_parse_error_does_not_invent_command_requests(self):
        row=self.record('{"analysis":"x","commands":[{"keystrokes":"ls\\n"}]}','private thought')
        blocks=self.extract(row)
        self.assertEqual([b['structural_type'] for b in blocks],['thinking','assistant_text'])

    def test_completion_is_only_a_declaration(self):
        row=self.record(json.dumps(dict(analysis='done',plan='',commands=[],task_complete=True)))
        block=next(b for b in self.extract(row) if b['structural_type']=='completion_declaration')
        self.assertTrue(block['declaration_is_not_evaluation_success'])
        self.assertEqual(block['semantic_status'],'not_assessed')

    def test_empty_commands_and_false_completion_create_no_requests(self):
        row=self.record(json.dumps(dict(analysis='x',plan='y',commands=[],task_complete=False)))
        kinds={b['structural_type'] for b in self.extract(row)}
        self.assertNotIn('command_request',kinds)
        self.assertNotIn('completion_declaration',kinds)

    def test_nested_completion_text_is_not_a_declaration(self):
        row=self.record(json.dumps(dict(analysis='Example: {"task_complete":true}',plan='y',commands=[])))
        self.assertNotIn('completion_declaration',{b['structural_type'] for b in self.extract(row)})

    def test_feedback_like_text_in_model_field_is_not_harness_feedback(self):
        row=self.record(json.dumps(dict(analysis='Previous response had warnings:\nnot a harness message',plan='y',commands=[])))
        self.assertNotIn('parser_warning',{b['structural_type'] for b in self.extract(row)})

    def test_following_feedback_match_flag_is_not_a_rule_input(self):
        row=self.record(json.dumps(dict(analysis='x',plan='y',commands=[])))
        before=self.extract(row)
        altered=copy.deepcopy(row)
        altered['parsed']['replay_matches_observed_feedback']=False
        altered['outcome']='future-format-retry'
        altered['labels']=[dict(kind='invented_future_label')]
        self.assertEqual(before,self.extract(altered))

    def test_raw_body_fragments_are_lossless_nonoverlapping(self):
        row=self.record('Preface\n'+json.dumps(dict(analysis='x\ny',plan='p',commands=[dict(keystrokes='ls\n',duration=.1)]))+'\n')
        blocks=[b for b in self.extract(row) if b['source']['pointer']=='/original/content']
        self.assertEqual(''.join(b['content'] for b in blocks),row['original']['content'])
        for a,b in zip(blocks,blocks[1:]):
            self.assertEqual(a['source']['character_span'][1],b['source']['character_span'][0])

    def test_parser_repair_uses_projection_without_fake_raw_spans(self):
        row=self.record('{"analysis":"inspect","plan":"read","commands":[]')
        self.assertFalse(row['parsed']['error'])
        blocks=self.extract(row)
        self.assertEqual(blocks[0]['structural_type'],'unmapped_wire_container')
        analysis=next(b for b in blocks if b['structural_type']=='analysis')
        self.assertEqual(analysis['layer'],'parser_projection')
        self.assertIsNone(analysis['source']['character_span'])
        self.assertTrue(analysis['no_raw_wire_character_span'])

    def test_session_audit_has_no_invented_step(self):
        record=dict(record_type='execution_event',original=dict(event='command',exit_code=0),source=dict(line=1))
        block=StructureAdapter(self.rules,self.parser).record_blocks(self.item,record,1,'')[0]
        self.assertEqual(block['layer'],'session_audit')
        self.assertIsNone(block['source']['step'])
        self.assertIsNone(block['available_through_step'])

    def test_progress_operation_is_a_tool_call_subtype(self):
        item=dict(family='deepswe',trajectory_id='deepswe/test',trace_path='trace.jsonl')
        record=dict(record_type='message',turn=1,message_index=1,role='assistant',original=dict(role='assistant',
            content=[dict(type='tool_use',id='c1',name='TaskUpdate',input=dict(taskId='1',status='completed'))]))
        blocks=StructureAdapter(self.rules,self.parser).record_blocks(item,record,1,'')
        self.assertEqual(len(blocks),1)
        self.assertEqual(blocks[0]['structural_type'],'progress_operation')
        self.assertEqual(blocks[0]['also_types'],['tool_call'])

    def test_frozen_loader_does_not_write_into_source_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            paths=['methodology/label_agent_trajectories.py','methodology/frozen/terminus_json_plain_parser.py']
            entries=[]
            for name in paths:
                data=(BUNDLE/name).read_bytes()
                path=root/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(data)
                entries.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest()))
            (root/'MANIFEST.json').write_text(json.dumps(dict(files=entries)))
            load_rules(root)
            self.assertFalse(list(root.rglob('*.pyc')))


if __name__=='__main__':
    unittest.main()
