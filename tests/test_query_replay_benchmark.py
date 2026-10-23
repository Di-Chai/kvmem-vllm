"""Freeze shared public inputs and compare full costs without claiming accuracy."""
import importlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
bench=importlib.import_module('benchmark_query_replay')


def test_performance_inputs_are_staggered_frozen_and_identical_between_arms(tmp_path,monkeypatch):
    from test_ruler_execution import profile as make_profile
    from ruler_execution import OPTIMIZED
    repo=tmp_path/'repo';(repo/'scripts').mkdir(parents=True)
    shutil.copy2(Path(bench.__file__),repo/'scripts/benchmark_query_replay.py')
    source=tmp_path/'source';(source/'inputs').mkdir(parents=True)
    profile=make_profile.__wrapped__();bench.save(source/'profile.json',profile)
    rows=[]
    for i in range(4):
        p=source/'inputs'/f'case{i}.i32'
        np.arange(131072,dtype='<i4').tofile(p)
        rows.append(dict(id=f'case{i}',served_sha256=bench.sha(p),served_query=[130000,130042],
                         question='public question'))
    fixture=dict(profile_sha256=bench.sha(source/'profile.json'),rows=rows,
                 input_directory=str(source/'inputs'),tokenizer_sha256='tokenizer',execution_profile=OPTIMIZED)
    bench.save(source/'fixture.json',fixture)
    bench.save(source/'dataset-summary.json',dict(fixture_sha256=bench.sha(source/'fixture.json')))
    bench.save(source/'runtime-proof.json',dict(files={}))
    monkeypatch.setattr(bench.subprocess,'check_output',lambda *args,**kwargs:'head\n')
    root=tmp_path/'benchmark'
    bench.prepare(root,source,repo)
    fixtures=[bench.read(root/arm/'fixture.json') for arm,_,_ in bench.ARMS]
    assert fixtures[0]==fixtures[1]==fixtures[2]
    assert fixtures[0]['performance_only'] and not fixtures[0]['accuracy_claim']
    assert [r['served_query'][0] for r in fixtures[0]['rows']]==[65536,81920,98304,114688]
    for row in fixtures[0]['rows']:
        tokens=np.fromfile(row['input_path'],dtype='<i4')
        assert tokens[row['served_query'][0]:row['served_query'][1]].tolist()==list(range(130000,130042))
        assert bench.sha(Path(row['input_path']))==row['served_sha256']
    profiles=[bench.read(root/arm/'profile.json') for arm,_,_ in bench.ARMS]
    for profile,(arm,backend,cobatch) in zip(profiles,bench.ARMS):
        assert profile['kvmem']['query_replay_backend']==backend
        assert profile['kvmem']['query_replay_cobatch'] is cobatch
        profile['kvmem'].pop('query_replay_backend');profile['kvmem'].pop('query_replay_cobatch')
    assert profiles[0]==profiles[1]==profiles[2]
    assert not (root/'baseline/run').exists()  # Freeze cannot load a model or acquire GPU work.


def test_comparison_reports_output_and_selection_changes_separately(tmp_path):
    for index,(name,backend,cobatch) in enumerate(bench.ARMS):
        arm=tmp_path/name
        bench.save(arm/'status.json',dict(status='completed',cold_prepare_seconds=20))
        bench.save(arm/'cleanup.json',dict(remaining_owned_groups={},gpu_processes=[]))
        bench.save(arm/'run/answers/completed.json',dict(seconds=100-10*index,
            input_tokens_per_campaign_second=1000,input_tokens_per_all_first_tokens_second=1200,
            execution_counts={backend:1,'cobatch':int(cobatch)},
            results=[dict(id='case',output_sha256='different' if cobatch else 'same')]))
        bench.save(arm/'run/answers/case/evidence.json',dict(query=dict(
            selected_history_ranges=[[32 if cobatch else 0,32]])))
        p=arm/'run/server/capture/async_refresh.jsonl';p.parent.mkdir(parents=True)
        p.write_text(json.dumps(dict(stage='query_cobatch' if cobatch else 'query',host_seconds=.2))+'\n')
    result=bench.analyze(tmp_path)
    assert result['compiled']['outputs_identical_to_baseline']
    assert not result['cobatch']['outputs_identical_to_baseline']
    assert not result['cobatch']['selected_history_identical_to_baseline']
    assert result['cobatch']['standalone_query_host_ms_median'] is None
    assert result['cobatch']['campaign_speedup_percent']==25
    report=bench.read(tmp_path/'comparison.json')
    assert not report['accuracy_claim'] and report['repeats']==1
