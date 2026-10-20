"""Read-only, low-frequency MTP counters alongside a frozen C8 experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import urlopen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--port',type=int,default=18726)
    args=parser.parse_args()
    root=args.root.resolve()
    definition=Path(__file__).read_bytes()
    (root/'metrics-observer.json').write_text(json.dumps(dict(
        script_sha256=hashlib.sha256(definition).hexdigest(),interval_seconds=5,
        scope='Independent read-only /metrics sampling, same cadence in all arms. '
              'Counters cover observed subwindows, not unobserved final steps.'),indent=2)+'\n')
    pattern=re.compile(r'^(vllm:spec_decode_num_(?:drafts|draft_tokens|accepted_tokens)_total)(?:\{.*\})?\s+(\S+)')
    while True:
        state=json.loads((root/'status.json').read_text())
        if state['status'] in ('completed','failed_attempt_preserved'):
            return
        if state['status']=='running':
            started=time.monotonic()
            try:
                with urlopen(f'http://127.0.0.1:{args.port}/metrics',timeout=2) as response:
                    raw=response.read().decode()
                counters={}
                for line in raw.splitlines():
                    match=pattern.match(line)
                    if match:
                        counters[match[1]]=counters.get(match[1],0.)+float(match[2])
                if counters:
                    with (root/(state['job']+'-mtp-counters.jsonl')).open('a') as stream:
                        stream.write(json.dumps(dict(started_at=started,finished_at=time.monotonic(),
                            unix=time.time(),counters=counters))+'\n')
            except OSError:
                pass  # loading and shutdown are expected; never restart an arm
        time.sleep(5)


if __name__=='__main__':
    main()
