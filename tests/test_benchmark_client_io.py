"""Bookkeeping equivalence and bounded I/O,without a GPU or model import."""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
try:
    io = importlib.import_module('benchmark_client_io')
    protocol = importlib.import_module('graph_v2_protocol')
finally:
    sys.path.pop(0)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def line(value):
    return (json.dumps(value,ensure_ascii=False)+'\n').encode()


def test_progress_matches_independent_disk_scan_without_rereading_old_results(tmp_path,monkeypatch):
    rows=[dict(id=name) for name in ['a','b','c']]
    fields={'id','status','usage','prompt_tokens','error'}
    initial=dict(id='c',status='completed',usage=dict(completion_tokens=7),graph_v2=dict(scores=[1]*100))
    save(tmp_path/'c/result.json',initial)
    progress=io.CompactResults(rows,tmp_path,fields)
    # An already-loaded result cannot be reopened merely to update progress.
    original=Path.read_text
    reads=[]
    def tracked(path,*args,**kwargs):
        reads.append(path)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',tracked)
    for record in [dict(id='b',status='error',error='retained technical error'),
                   dict(id='a',status='completed',prompt_tokens=90,usage=dict(completion_tokens=9))]:
        save(tmp_path/record['id']/'result.json',record)
        progress.put(record)
        assert not reads
        expected=[]
        for row in rows:
            path=tmp_path/row['id']/'result.json'
            if path.exists():
                full=json.loads(original(path))
                expected.append({k:v for k,v in full.items() if k in fields})
        assert progress.snapshot()==expected
        assert not reads
    assert 'graph_v2' not in progress.snapshot()[-1]


def test_compact_progress_owns_metadata_and_preserves_resume_order(tmp_path):
    rows=[dict(id='a'),dict(id='b')]
    progress=io.CompactResults(rows,tmp_path,('usage',))
    result=dict(id='b',status='completed',usage=dict(completion_tokens=4))
    progress.put(result);result['usage']['completion_tokens']=999
    copy=progress.snapshot();copy[0]['usage']['completion_tokens']=888
    assert progress.snapshot()[0]['usage']['completion_tokens']==4
    save(tmp_path/'b/result.json',dict(id='b',status='completed',usage=dict(completion_tokens=4)))
    resumed=io.CompactResults(rows,tmp_path,('usage',))
    assert resumed.snapshot()==progress.snapshot()
    with pytest.raises(ValueError):progress.put(dict(id='foreign',status='completed'))
    with pytest.raises(ValueError):io.CompactResults([dict(id='a'),dict(id='a')],tmp_path,())
    save(tmp_path/'a/result.json',dict(id='foreign',status='completed'))
    with pytest.raises(ValueError,match='identity'):io.CompactResults(rows,tmp_path,())


def test_capture_slice_parses_only_new_rows_and_reuses_one_snapshot(tmp_path):
    path=tmp_path/'capture.jsonl';old=b'old data must never be parsed\n'*1000
    path.write_bytes(old);cursor=io.CaptureSlice([path])
    appended=line(dict(request_id='current',value=1))+line(dict(request_id='other',value=2))
    with path.open('ab') as stream:stream.write(appended)
    result=cursor.read(path)
    assert list(result)==[dict(request_id='current',value=1),dict(request_id='other',value=2)]
    assert cursor.bytes_read==len(appended) and cursor.records_read==2
    with path.open('ab') as stream:stream.write(line(dict(request_id='later')))
    assert cursor.read(path) is result and cursor.bytes_read==len(appended)


def test_capture_slice_handles_prior_utf8_fragment_and_trailing_foreign_fragment(tmp_path):
    path=tmp_path/'capture.jsonl';path.write_bytes(b'{"request_id":"old","text":"\xe4')
    cursor=io.CaptureSlice([path]);wanted=dict(request_id='current',text='中文')
    with path.open('ab') as stream:stream.write(b'\xbd\xa0"}\n'+line(wanted)+b'{"request_id":"foreign"')
    assert cursor.read(path)==(wanted,)


def test_missing_or_unregistered_capture_paths_do_not_trigger_a_full_scan(tmp_path):
    path=tmp_path/'new.jsonl';missing=tmp_path/'absent.jsonl'
    cursor=io.CaptureSlice([path,missing]);wanted=dict(request_id='new')
    path.write_bytes(line(wanted))
    assert cursor.read(path)==(wanted,) and cursor.read(missing)==()
    with pytest.raises(ValueError):cursor.read(tmp_path/'not-marked.jsonl')
    other=io.CaptureSlice([path]);path.unlink()
    with pytest.raises(RuntimeError,match='disappeared'):other.read(path)


def test_rotated_truncated_or_rewritten_capture_is_rejected(tmp_path):
    path=tmp_path/'events.jsonl';prefix=line(dict(request_id='old',payload='x'*100))
    path.write_bytes(prefix);cursor=io.CaptureSlice([path]);path.write_bytes(b'{}\n')
    with pytest.raises(RuntimeError,match='truncated'):cursor.read(path)
    path.write_bytes(prefix);cursor=io.CaptureSlice([path]);path.rename(tmp_path/'old.jsonl');path.write_bytes(prefix+line({}))
    with pytest.raises(RuntimeError,match='rotated'):cursor.read(path)
    cursor=io.CaptureSlice([path]);path.write_bytes(b'R'*len(prefix)+line({})+line({}))
    with pytest.raises(RuntimeError,match='prefix changed'):cursor.read(path)


def test_malformed_new_complete_line_is_not_silently_ignored(tmp_path):
    path=tmp_path/'events.jsonl';path.write_bytes(line({}));cursor=io.CaptureSlice([path])
    with path.open('ab') as stream:stream.write(b'not valid json\n')
    with pytest.raises(json.JSONDecodeError):cursor.read(path)


def test_overlapping_requests_keep_separate_boundaries(tmp_path):
    path=tmp_path/'events.jsonl';path.write_bytes(line(dict(request_id='old')))
    first=io.CaptureSlice([path]);a=dict(request_id='a-0')
    with path.open('ab') as stream:stream.write(line(a))
    second=io.CaptureSlice([path]);b=dict(request_id='b-0');a2=dict(request_id='a-0',phase='done')
    with path.open('ab') as stream:stream.write(line(b)+line(a2))
    assert first.read(path)==(a,b,a2) and second.read(path)==(b,a2)


def test_graph_audit_uses_scoped_reader_with_original_response_matching(tmp_path,monkeypatch):
    refresh=tmp_path/'bounded_refresh.jsonl';graph=tmp_path/'block_graph.jsonl'
    wanted=dict(request_id='r-0',phase='query');equal=dict(request_id='r',phase='pressure');foreign=dict(request_id='other')
    def forbidden(*args):raise AssertionError('whole-log reader used')
    def matches(response,event):
        rid=event['request_id'];return rid==response['response_id'] or rid.startswith(response['response_id']+'-')
    monkeypatch.setitem(sys.modules,'qualify_block_graph',SimpleNamespace(native_matches=matches,records=forbidden))
    seen={}
    def audit(events,graphs,profile,arm,query,**kwargs):
        seen.update(events=events,graphs=graphs,kwargs=kwargs);return 'audited'
    monkeypatch.setattr(protocol,'audit_events',audit);monkeypatch.setattr(protocol,'infer_arm',lambda profile:'C')
    reads=[]
    def scoped(path):
        reads.append(path);return [wanted,foreign] if path==refresh else [equal,foreign]
    assert protocol.audit_case(tmp_path,'r',{},[1,2],require_pressure=True,event_reader=scoped)=='audited'
    assert reads==[refresh,graph] and seen==dict(events=[wanted],graphs=[equal],kwargs=dict(require_pressure=True))
