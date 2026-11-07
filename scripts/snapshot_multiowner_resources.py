"""One read-only host placement snapshot of an owned, active measured worker."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time


def snapshot(root, output):
    assert not output.exists(), 'Preserve the first placement snapshot'
    work=json.loads((root/'run/server/capture/public_work.json').read_text())
    process=json.loads((root/'process.json').read_text())
    pid=work['pid']
    if os.getpgid(pid)!=process['server_pid']:
        raise ValueError('Worker is not in the measured server process group')
    status=dict(line.split(':',1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
    nodes=defaultdict(int); anonymous=defaultdict(int); policies=defaultdict(int)
    for line in Path(f'/proc/{pid}/numa_maps').read_text().splitlines():
        fields=line.split(); policies[fields[1]]+=1
        values={s.split('=',1)[0]:s.split('=',1)[1] for s in fields[2:] if '=' in s}
        page=int(values.get('kernelpagesize_kB',4))*1024
        for key,value in values.items():
            if key.startswith('N') and key[1:].isdigit():
                nodes[key]+=int(value)*page
                if 'anon' in values:anonymous[key]+=int(value)*page
    meminfo={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
    result=dict(at=time.time(),pid=pid,root=str(root),completed_prefill=work['completed']['public_prefill_tokens'],
        process={k:status[k].strip() for k in ('Threads','VmRSS','VmHWM','Cpus_allowed_list','Mems_allowed_list')},
        node_resident_bytes=dict(nodes),anonymous_node_bytes=dict(anonymous),policies=dict(policies),
        system_memory={k:meminfo[k] for k in ('MemTotal','MemAvailable','SwapTotal','SwapFree')},
        scope='Single /proc read; no affinity, memory policy, scheduling or cache mutation. Does not prove NUMA causality.')
    with output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();snapshot(a.root.resolve(),a.output.resolve())
