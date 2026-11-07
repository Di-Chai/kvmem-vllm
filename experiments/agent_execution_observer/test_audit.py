"""Review normalization must not mutate native output or invent boundaries."""
import unittest
import hashlib
import json
from pathlib import Path
import tempfile

from audit import audit_run, diagnostics, normalize_boundaries
from observer import Memory, digest, extract_step, request_for, save


class ReviewInvariants(unittest.TestCase):
    def test_only_v1_unresolved_to_existing_is_normalized(self):
        rows = [dict(window_id='a', step=i + 1, stage_id=sid, stage_boundary=boundary)
                for i, (sid, boundary) in enumerate([
                    ('S001', False), (None, None), ('S001', True),
                    (None, None), ('S002', True)])]
        normalized = normalize_boundaries(rows)
        self.assertIsNone(normalized[2]['stage_boundary'])
        self.assertTrue(rows[2]['stage_boundary'])
        self.assertTrue(normalized[4]['stage_boundary'])

    def test_windows_do_not_share_normalization_state(self):
        rows = [dict(window_id='a', stage_id=None, stage_boundary=None),
                dict(window_id='b', stage_id='S001', stage_boundary=False)]
        self.assertFalse(normalize_boundaries(rows)[1]['stage_boundary'])

    def test_format_diagnostics_are_not_semantic_accuracy(self):
        rows = [dict(diagnostics={'stage_link':dict(label_mass=.01,argmax_is_label=False)}),
                dict(diagnostics={'stage_link':dict(label_mass=.99,argmax_is_label=True)})]
        result = diagnostics(rows)['stage_link']
        self.assertEqual(result['argmax_is_label_false'],1)
        self.assertEqual(result['label_mass']['mean'],.5)
        self.assertNotIn('accuracy',result)


class AuditTampering(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / 'bundle'
        self.bundle.mkdir()
        self.results = self.root / 'results'
        self.results.mkdir()
        self.dataset = self.root / 'dataset.jsonl'
        item = dict(trajectory_id='t', task_id='task', seed=1001, trace_path='trace.jsonl')
        raw = [dict(turn=0, labels=[dict(kind='task_instruction', path='/original/content/0/text')],
                    original=dict(role='user',content=[dict(type='text',text='Fix parser')])),
               dict(record_type='message', turn=1, message_index=1,
                    original=dict(role='assistant', content=[dict(type='text',text='Read parser'),
                        dict(type='tool_use',id='c1',name='Read',input=dict(file_path='parser.py'))])),
               dict(record_type='message', turn=1, message_index=2,
                    original=dict(role='user', content=[dict(type='tool_result',tool_use_id='c1',content='Parser source')]))]
        (self.bundle / 'trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in raw))
        save(self.bundle / 'index.json',[item])
        step = extract_step(item,1,list(enumerate(raw,1))[1:])
        step.update(window_id='w',task_instruction='Fix parser')
        self.dataset.write_text(json.dumps(step)+'\n')
        save(self.root / 'manifest.json',dict(dataset_sha256=digest(self.dataset),
             source_hashes={p.name:digest(p) for p in self.bundle.iterdir()},
             windows=[dict(left_censored=False)]))
        save(self.results / 'run.json',dict(dataset_sha256=digest(self.dataset),
             model='observer',mode='memory',observer_sha256=digest(Path(__file__).with_name('observer.py'))))
        memory = Memory()
        payload = request_for(step,memory,'observer')
        choices = dict(activity='code_understanding',stage_link='new_stage',outcome='in_progress',evidence='e1')
        response = dict(object='structured_decision',model='observer',answers={},diagnostics={},
                        usage=dict(input_tokens=100,output_tokens=4))
        for qid,question in payload['questions'].items():
            response['answers'][qid] = dict(choice=choices[qid],confidence=.9,
                probabilities={label:float(label==choices[qid]) for label in question['criteria']})
            response['diagnostics'][qid] = dict(label_mass=.9,argmax_is_label=True)
        ident=hashlib.sha256(b'w').hexdigest()[:12]+'-0001'
        self.request = self.results / 'calls' / f'{ident}.request.json'
        save(self.request,payload)
        save(self.results / 'calls' / f'{ident}.response.json',response)
        prediction=memory.apply(step,response)
        prediction.update(call_id=ident,seconds=1.,usage=response['usage'],
                          diagnostics=response['diagnostics'],native_answers=response['answers'])
        (self.results / 'decisions.jsonl').write_text(json.dumps(prediction)+'\n')
        save(self.results / 'summary.json',dict(status='completed',steps=1,input_tokens=100,
             total_seconds=1.,stage_boundaries=0,label_reads=4))
        save(self.results / 'memory.json',{'w':memory.view()})

    def run_audit(self):
        return audit_run(self.bundle,self.dataset,self.results,self.root/'audit')

    def test_exact_raw_request_reconstruction_passes(self):
        self.assertEqual(self.run_audit()['status'],'passed')

    def test_future_memory_is_rejected(self):
        payload=json.loads(self.request.read_text())
        payload['state']['execution_memory']['recent_steps']=[dict(step=2,excerpt='Future')]
        save(self.request,payload)
        with self.assertRaisesRegex(ValueError,'Request contains different'):
            self.run_audit()

    def test_exporter_labels_cannot_be_added_to_state(self):
        payload=json.loads(self.request.read_text())
        payload['state']['exporter_outcome']='success'
        save(self.request,payload)
        with self.assertRaisesRegex(ValueError,'Request contains different'):
            self.run_audit()

    def test_option_order_change_is_rejected(self):
        payload=json.loads(self.request.read_text())
        criteria=payload['questions']['activity']['criteria']
        payload['questions']['activity']['criteria']=dict(reversed(list(criteria.items())))
        save(self.request,payload)
        with self.assertRaisesRegex(ValueError,'option order'):
            self.run_audit()

    def test_changed_source_hash_is_rejected(self):
        with (self.bundle/'trace.jsonl').open('a') as stream:
            stream.write('{}\n')
        with self.assertRaisesRegex(ValueError,'Source hash mismatch'):
            self.run_audit()

    def test_tool_return_from_another_step_is_rejected(self):
        source=self.bundle/'trace.jsonl'
        rows=[json.loads(line) for line in source.read_text().splitlines()]
        rows[-1]['original']['content'][0]['tool_use_id']='future-call'
        source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        manifest=json.loads((self.root/'manifest.json').read_text())
        manifest['source_hashes']['trace.jsonl']=digest(source)
        save(self.root/'manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'Tool result refers to another Step'):
            self.run_audit()


if __name__ == '__main__':
    unittest.main()
