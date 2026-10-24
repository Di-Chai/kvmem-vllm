import json
from diagnostics.dsh_cloud.lifecycle_probe import artifact_gc_evidence

def fixture(tmp_path, *, source='verified_dsh_write_artifact', live=True, namespace=None):
 owner='owner';root=tmp_path/owner
 plan=dict(source=source,namespace=namespace or [owner,owner],span_id='real',relative_path='app.js',source_begin=32,source_end=96)
 state=dict(plan=plan,flushed=True,complete=True,observed=True,error=None,archive_request_id='native-id',
 candidate_handles=[[1,2]],remaining_source_records=0,window_source_records=0,
 events=[dict(kind='source_record_released',handle=[1,2],history_begin=64,during_live_scope=live)])
 for i in (1,2):
  turn=root/f'{i:05d}';turn.mkdir(parents=True)
  (turn/'accepted.json').write_text('{}')
  (turn/'artifact-gc-state.json').write_text(json.dumps(dict(results=[[state]])))
 return artifact_gc_evidence(tmp_path)

def test_candidate_release_is_deduplicated_and_later_decode_is_reported(tmp_path):
 result=fixture(tmp_path)
 assert result['live_source_cpu_completions']==1
 assert result['later_committed_model_replies']==1
 assert result['task_continuation_after_release_proven']
 assert result['source_spans'][0]['working_window_source_records']==0

def test_cleanup_events_do_not_prove_semantic_gc(tmp_path):
 result=fixture(tmp_path,live=False)
 assert result['semantic_producer_present']
 assert not result['real_semantic_candidate_retirement_proven']

def test_synthetic_fixture_does_not_become_real_semantic_evidence(tmp_path):
 result=fixture(tmp_path,source='synthetic_gc_throughput_fixture')
 assert not result['semantic_producer_present']
 assert result['live_source_cpu_completions']==0

def test_foreign_namespace_does_not_become_owned_evidence(tmp_path):
 result=fixture(tmp_path,namespace=['foreign','foreign'])
 assert not result['semantic_producer_present']
