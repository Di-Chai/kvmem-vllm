"""Verify a frozen remote manifest before regenerating local derived files."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):
            value.update(chunk)
    return value.hexdigest()


def main(root):
    manifest=json.loads((root/'manifest.sha256.json').read_text())
    verified=[];remote_only=[];errors=[]
    for entry in manifest['files']:
        relative=Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Manifest path escapes the experiment directory')
        path=root/relative
        if not path.exists() and path.suffix in ('.pt','.bin'):
            remote_only.append(entry)
            continue
        if not path.is_file():
            errors.append(dict(path=entry['path'],error='missing file'))
        elif path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']:
            errors.append(dict(path=entry['path'],error='size or SHA-256 mismatch'))
        else:
            verified.append(entry)
    result=dict(verified_at=time.time(),passed=not errors,
        verified_files=len(verified),verified_bytes=sum(r['bytes'] for r in verified),
        remote_only_files=remote_only,errors=errors,
        scope='Frozen remote files before local analysis regeneration. Native KV archives remain on the experiment server; their recorded hashes are retained.')
    (root/'download-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='remote_only_files'}))
    if errors:raise RuntimeError('Downloaded raw evidence differs from the frozen manifest')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
