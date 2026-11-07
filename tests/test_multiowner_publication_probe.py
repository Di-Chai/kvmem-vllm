from types import SimpleNamespace as NS
import json
import multiowner_publication_probe as module


def test_bounded_capture_keeps_all_owners_and_maintenance(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'scheduled_rows',lambda rows:rows)
    now=[0.];inner=NS(is_running=False)
    inner.start=lambda:setattr(inner,'is_running',True)
    inner.stop=lambda:setattr(inner,'is_running',False)
    gate=module.MultiOwnerProfiler(inner,dict(begin_tokens=96000,observe_seconds=8),tmp_path,clock=lambda:now[0])
    gate.start();row=lambda rid,n:dict(request_id=rid,computed=n,tokens=3072,prefill=True)
    gate.before_step([row('a',0),row('b',0)])
    assert not gate.is_running and gate.owner=='a'
    gate.before_step([row('b',97000)]);assert not gate.is_running
    gate.before_step([row('a',97000),row('b',99000)])
    assert gate.is_running and len(gate.rows[0]['owners'])==2
    now[0]=7;gate.before_step([]);assert gate.is_running
    now[0]=8;gate.before_step([])
    assert gate.done and not gate.is_running
    report=json.loads((tmp_path/'pressure-range.json').read_text())
    assert report['reason']=='bounded_time_window' and report['elapsed_seconds']==8
