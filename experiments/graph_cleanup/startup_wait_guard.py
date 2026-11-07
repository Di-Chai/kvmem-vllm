"""Resume an owned paused startup controller after slow storage reads finish."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from urllib.request import build_opener, ProxyHandler


def main(args):
    cmdline=Path(f'/proc/{args.controller}/cmdline')
    assert cmdline.exists() and b'continue_gpu.py' in cmdline.read_bytes()
    event=Path('startup-wait-extension.json')
    record=json.loads(event.read_text())
    opener=build_opener(ProxyHandler({}))
    ready=False
    try:
        for _ in range(300):
            if not Path(f'/proc/{args.api}').exists():break
            try:
                with opener.open('http://127.0.0.1:18996/health',timeout=2) as response:
                    ready=response.status==200
                if ready:break
            except Exception:pass
            time.sleep(2)
    finally:
        # Identity check protects against accidentally signaling a reused PID.
        if cmdline.exists() and b'continue_gpu.py' in cmdline.read_bytes():
            os.kill(args.controller,signal.SIGCONT)
        record.update(resumed=time.time(),api_ready=ready)
        event.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--controller',type=int,required=True)
    p.add_argument('--api',type=int,required=True)
    main(p.parse_args())
