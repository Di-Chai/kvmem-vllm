"""Stream an immutable checkpoint of completed serving cells to stdout."""
import io
import json
from pathlib import Path
import re
import sys
import tarfile


root=Path.cwd()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
    for relative in ('serving-r1','serving-r1/native-systemone'):
        base=root/relative
        if not (base/'summary.json').exists():continue
        rows=json.loads((base/'summary.json').read_text())
        for row in rows:
            name=row['name']
            if not re.fullmatch(r'l\d+(?:-o\d+)?-c\d+(?:-(?:observer|pair))?',name):
                raise ValueError('Unexpected cell path')
            archive.add(base/name,arcname=relative+'/'+name)
        # The summary is the captured list, never the concurrently updated file.
        content=(json.dumps(rows,indent=2)+'\n').encode()
        info=tarfile.TarInfo(relative+'/summary.json');info.size=len(content)
        archive.addfile(info,io.BytesIO(content))
        for name in ('design.json','input-token-sets.json','warmup.json','status.json'):
            if name=='status.json' and len(rows)!=(120 if relative=='serving-r1' else 20):continue
            path=base/name
            if path.exists():archive.add(path,arcname=relative+'/'+name)
