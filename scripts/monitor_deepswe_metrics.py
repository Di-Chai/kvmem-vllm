"""Read-only native counters for the ongoing DeepSWE run.

Counter differences exclude qualification and cached prompt tokens. This
monitor may start after inference, so its coverage is explicitly timestamped.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


NAMES={
    'vllm:num_preemptions_total', 'vllm:prompt_tokens_total',
    'vllm:prompt_tokens_cached_total', 'vllm:generation_tokens_total',
    'vllm:request_prefill_kv_computed_tokens_sum',
    'vllm:request_prefill_time_seconds_sum', 'vllm:request_decode_time_seconds_sum',
    'vllm:request_queue_time_seconds_sum', 'vllm:spec_decode_num_draft_tokens_total',
    'vllm:spec_decode_num_accepted_tokens_total',
    'vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:kv_cache_usage_perc',
}


def run(root):
    config=json.loads((root/'profile.json').read_text())
    client=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    output=root/'native-metrics.jsonl'
    while True:
        status=json.loads((root/'run.json').read_text())['status']
        if status in ('completed','needs_attention'):return
        try:
            with client.open(f"http://127.0.0.1:{config['port']}/metrics",timeout=10) as response:
                text=response.read().decode()
            counters={}
            for line in text.splitlines():
                if line.startswith('#'):continue
                if not line.strip():continue
                name=line.split()[0].split('{',1)[0]
                if name in NAMES:
                    counters[name]=counters.get(name,0)+float(line.rsplit(' ',1)[1])
            archive=json.loads((root/'server/capture/archive.json').read_text())
            sample={'at':time.time(),'run_phase':status,'counters':counters,
                    'cpu_authority_bytes':archive.get('cpu_authority_bytes'),
                    'checkpoint_payload_bytes':archive.get('session_checkpoint_payload_bytes')}
            resources=root/'server/capture/scheduler_resources.json'
            if resources.exists():
                scheduler=json.loads(resources.read_text())
                sample['scheduler']={k:v for k,v in scheduler.items() if k in (
                    'at','requests','refresh_pending','connector_pending','authority_limit','cpu_history_admission',
                    'active_residency','shared_prefill','offline_policy')}
        except (OSError,urllib.error.URLError,ValueError) as error:
            sample={'at':time.time(),'error':repr(error)}
        with output.open('a') as stream:stream.write(json.dumps(sample)+'\n')
        time.sleep(10)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    run(parser.parse_args().root)
