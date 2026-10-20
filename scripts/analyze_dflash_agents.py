"""Audit frozen live-tool DFlash/MTP windows with their actual K/G and KV geometry."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def records(p):return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def analyze(root):
    m=read(root/'manifest.json')
    if read(root/'status.json')['status']!='completed':raise ValueError('All declared live windows must complete')
    qualification=Path(m['qualification'])
    if (sha(qualification/'manifest.json')!=m['qualification_sha256'] or
        sha(qualification/'qualification/results.json')!=m['qualification_result_sha256'] or
        read(qualification/'status.json')['status']!='passed'):
        raise ValueError('State qualification evidence changed or did not pass')
    for name,digest in m['source_sha256'].items():
        if sha(root/'frozen'/name)!=digest:raise ValueError('Frozen source changed: '+name)
    for name,digest in m['input_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Frozen input changed: '+name)
    values={};evidence={str(root/'manifest.json'):sha(root/'manifest.json')}
    jobs={name:root/name for name in m['order']}
    if m.get('reference_mtp'):
        reference=Path(m['reference_mtp']);original=read(reference.parent/'manifest.json')
        if read(reference.parent/'status.json')['status']!='completed':
            raise ValueError('Reused original comparison did not complete')
        for name,digest in original['source_sha256'].items():
            if sha(reference.parent/'frozen'/name)!=digest:
                raise ValueError('Reused MTP frozen source changed')
            if name.startswith('src/') and m['source_sha256'].get(name)!=digest:
                raise ValueError('Reused MTP serving source differs')
        for name,digest in original['input_sha256'].items():
            if sha(reference.parent/name)!=digest:raise ValueError('Reused MTP input changed')
        jobs={'mtp4':reference,**jobs}
    for name,job in jobs.items():
        state=read(job/'run.json');a=read(job/'analysis.json');p=read(job/'profile.json')
        if state['status']!='window_complete' or state.get('cleanup_errors'):
            raise ValueError('Original controller or cleanup failed: '+name)
        capture=job/'server/capture';g=read(capture/'cache-geometry.json');archive=read(capture/'archive.json')
        scheduler=read(capture/'scheduler_resources.json')
        kv=p['kvmem'];expected_bytes=(17 if p['draft_method']=='mtp' else 16)*4*256*2
        contract=(g['select_budget']==kv['select_budget'] and g['gen_budget']==kv['gen_budget']
            and g['historical_tokens']>=p['max_model_len'] and g['cpu_offload_bytes']==kv['cpu_read_cache_bytes']
            and g['cpu_archive_bytes_per_token']==expected_bytes
            and g['archive_codec']=='raw_fp8_e4m3fn_v_fp8_e4m3fn_v1'
            and archive['archive_backend']=='cpu' and not archive['repository_gc']
            and archive['disk_read_bytes']==archive['disk_write_bytes']==0)
        if not contract or a['historical_view']['budget_violations']:
            raise ValueError('Actual profile, geometry, history budget or SSD contract differs')
        actions=records(capture/'action_attention.jsonl');graphs=records(capture/'block_graph.jsonl')
        config=read(capture/'action_config.json')
        if not all(config[k] for k in ('native_decode_commit','download_pool','fused_scoring')):
            raise ValueError('Attention implementation switches differ')
        if not graphs or any(r.get('backend')!='cpp_v2' for r in graphs):
            raise ValueError('Native V2 propagation did not execute')
        pool=actions[-1]['download_pool_total']
        if pool['exhaustions'] or pool['quarantined_slots'] or pool['allocated_pinned_bytes']>pool['max_pinned_bytes']:
            raise ValueError('Pinned pool failed or exceeded capacity')
        duration=state['window_deadline']-state['model_ready_at']
        events=records(job/'bridge/requests.jsonl')
        finished=[r for r in events if r.get('event')=='finish' and state['model_ready_at']<=r['at']<=state['window_deadline']]
        if sum(r['usage']['completion_tokens'] for r in finished)!=a['finished_turn_output_tokens']:
            raise ValueError('Completed-response token accounting differs')
        stream_counts=Counter()
        for path in (job/'cases').glob('*/claude.stream.jsonl'):
            for r in records(path):
                if r.get('type') in ('assistant','user'):
                    for block in (r.get('message') or {}).get('content',[]):
                        if isinstance(block,dict):stream_counts[block.get('type')]+=1
        # Retain compact statistics; the original analysis includes complete
        # snapshot page arrays and remains unchanged with its original hashes.
        compact={k:v for k,v in a.items() if k!='historical_view'}
        hv=a['historical_view']
        compact['history_audit']=dict(actual_profile_and_geometry_match=contract,
            logical_request_limit=p['max_model_len'],storage_history_capacity=g['historical_tokens'],
            cpu_archive_bytes_per_token=expected_bytes,budget_violations=hv['budget_violations'],
            consistent_publication_snapshots=hv['consistent_publication_snapshots'],
            maximum_coherently_compressed_idle_users=hv['maximum_coherently_compressed_users'],
            legacy_capacity_contract_applicable=False,
            scope='Legacy helper requires K64/G32/MTP bytes; its contract_matches=false is not this experiment contract. '
                'Validate actual K36/G16 and 16/17 archived layers here. Idle-only snapshots undercount active users. '
                'Storage capacity exceeds the stricter bridge/model logical-request cap; no truncation is introduced.')
        compact.update(formal_seconds=duration,cold_preparation_seconds=state['model_ready_at']-state['started_at'],
            execution=dict(draft_method=p['draft_method'],draft_tokens=p['draft_tokens'],
                execution_mode=p['execution_mode'],native_graph_mode=g['graph_mode'],
                gpu_memory_utilization=p['gpu_memory_utilization']),
            native_gpu_kv=dict(capacity_bytes=g['capacity_bytes'],native_blocks=g['native_blocks'],
                groups=len(g['groups']),recurrent_groups=sum(x['recurrent'] for x in g['groups']),
                reservation_rejections_lifetime=scheduler['reservation_rejections'],
                authority_rejections_lifetime=scheduler['authority_rejections'],
                scope='Final scheduler counters include initialization; rejection count is scheduling attempts, not failed requests.'),
            complete_response_output_tokens_per_second=a['finished_turn_output_tokens']/duration,
            complete_response_rate_per_second=len(finished)/duration,
            agent_stream_content_blocks=dict(stream_counts),
            agent_stream_scope='Whole owned attempt through shutdown; includes boundary tool results and is not the formal tool-call count.',
            modes_lifetime=dict(native_decode_calls=max(r['decode_commit_native_calls'] for r in actions),
                python_decode_calls=max(r['decode_commit_python_calls'] for r in actions),
                scorer_backends=actions[-1]['scoring_backend_counts'],native_graph_calls=len(graphs),download_pool=pool),
            startup_autotune_oom_fallback_messages=(job/'server.log').read_text().count('OOM detected, falling back to default tactic'),
            original_controller_status=state['status'],cleanup_errors=state['cleanup_errors'],
            native_qk_reuse=False,task_quality_qualified=False)
        values[name]=compact
        for path in (job/'profile.json',job/'manifest.json',job/'run.json',job/'analysis.json',
                     job/'bridge/requests.jsonl',job/'resources.jsonl',capture/'archive.json',capture/'cache-geometry.json',
                     capture/'scheduler_resources.json',capture/'action_attention.jsonl',capture/'block_graph.jsonl'):
            evidence[str(path)]=sha(path)
    left,right=values['mtp4'],values['dflash7']
    profiles={name:read(job/'profile.json') for name,job in jobs.items()}
    mtp,df=profiles['mtp4'],profiles['dflash7']
    differences={k:[mtp.get(k),df.get(k)] for k in mtp.keys()|df.keys() if mtp.get(k)!=df.get(k)}
    allowed={'draft_method','draft_tokens','draft_model','execution_mode',
             'max_num_batched_tokens','experimental_dflash_query_planning'}
    if m.get('reference_mtp'):allowed.add('gpu_memory_utilization')
    if set(differences)-allowed or mtp['kvmem']!=df['kvmem']:
        raise ValueError('Unmatched target policy or unqualified deployment difference')
    if ((mtp['draft_method'],mtp['draft_tokens'],mtp['execution_mode'])!=('mtp',4,'compiled_graph') or
        (df['draft_method'],df['draft_tokens'],df['execution_mode'])!=('dflash',7,'full_graph')):
        raise ValueError('Wrong declared speculative methods or execution modes')
    scope=m['scope']
    if m.get('reference_mtp'):
        plan=read(root/'targeted-plan.json')
        original_df=read(Path(plan['reference_dflash'])/'profile.json')
        actual_diff={k:[original_df.get(k),df.get(k)] for k in original_df.keys()|df.keys()
                     if original_df.get(k)!=df.get(k)}
        if actual_diff!=plan['profile_diff'] or set(actual_diff)!={'gpu_memory_utilization'}:
            raise ValueError('Targeted DFlash profile differs beyond its declared GPU-memory change')
        scope=scope.replace('No historical result is reused.','')+' '+plan['scope_correction']
        evidence[str(root/'targeted-plan.json')]=sha(root/'targeted-plan.json')
    return dict(kind='optimized_mtp_vs_experimental_dflash_live_tools_v1',analyzer_sha256=sha(Path(__file__)),
        measurements=values,observed_percent_change={k:100*(right[k]/left[k]-1) for k in
            ('complete_response_output_tokens_per_second','complete_response_rate_per_second')},
        profile_differences=differences,evidence_sha256=evidence,qualification=read(root.parent/'planned-state-proof.json'),
        scope=scope,limitation='One 300-second closed-loop live-tool window per deployment, eight workers; '
            'natural trajectories, task mixes and acceptance lengths differ. Native generated token rate includes '
            'unfinished requests; complete-response output rate counts only responses finished inside the formal window. '
            'No full first16 task-quality acceptance, pure-algorithm attribution or production DFlash default change. '
            'Cold timing includes first-use compilation/autotuning and order/cache effects. Lifetime mode counters include warmup.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous analyses')
    result=analyze(args.root.resolve());args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:dict(response_output_tps=v['complete_response_output_tokens_per_second'],
        native_generation_tps=v['whole_window']['generation_tokens_per_second'],turns=v['completed_model_turns'],
        tools=v['tool_calls'],acceptance_length=v['whole_window']['speculation']['mean_acceptance_length'])
        for name,v in result['measurements'].items()},indent=2))
