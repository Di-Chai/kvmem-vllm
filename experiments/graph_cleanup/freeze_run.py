"""Hash the finished run and existing model weights, outside timed workloads."""
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import time


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):h.update(block)
    return h.hexdigest()


root=Path.cwd()
assert json.loads((root/'pipeline-status.json').read_text())['stage']=='completed'
assert json.loads((root/'serving-r1/status.json').read_text())['cells']==120
assert json.loads((root/'serving-r1/native-systemone/status.json').read_text())['cells']==20
assert json.loads((root/'restored-service.json').read_text())['restored']
versions={}
for package in ('torch','vllm','flashinfer-python','transformers','safetensors','numpy'):
    try:versions[package]=metadata.version(package)
    except metadata.PackageNotFoundError:versions[package]=None
(root/'package-versions.json').write_text(json.dumps(versions,indent=2)+'\n')
weights=[]
initial_metadata=json.loads((root/'model-source-metadata.json').read_text())
for name,sha in initial_metadata['source_sha256'].items():
    assert digest(Path(name))==sha, 'Model configuration or inference implementation changed: '+name
original={r['name']:r for r in initial_metadata['model_files']}
for path in sorted(Path('/root/data/kvmem-vllm-gc/models/Qwen3.8-27B-GPTQ-Int4').glob('*.safetensors')):
    assert path.stat().st_size==original[path.name]['bytes'] and path.stat().st_mtime_ns==original[path.name]['mtime_ns'], 'Model shard changed during the experiment'
    sha=digest(path)
    assert path.stat().st_size==original[path.name]['bytes'] and path.stat().st_mtime_ns==original[path.name]['mtime_ns'], 'Model shard changed while hashing'
    weights.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha))
    print('Hashed',path.name,flush=True)
(root/'model-weight-sha256.json').write_text(json.dumps(weights,indent=2)+'\n')
tokenizer=Path('/root/data/recall-comparison-20261028-r1/assets/tokenizer')
tokenizer_files=[dict(path=str(path.relative_to(tokenizer)),bytes=path.stat().st_size,sha256=digest(path))
                 for path in sorted(tokenizer.rglob('*')) if path.is_file() and '.cache' not in path.parts]
(root/'tokenizer-sha256.json').write_text(json.dumps(dict(directory=str(tokenizer),files=tokenizer_files,
    scope='Final tokenizer files; exact token IDs and offsets used by the experiments are retained in prepared cases.'),indent=2)+'\n')
evidence=root/'frozen-evidence';evidence.mkdir(exist_ok=True)
if (root/'restored-service.log').exists():shutil.copy2(root/'restored-service.log',evidence/'restored-service-startup.log')
source=root/'experiments/graph_cleanup';frozen=root/'code';frozen.mkdir(exist_ok=True)
for path in source.iterdir():
    if path.is_file() and path.suffix in ('.py','.md'):shutil.copy2(path,frozen/path.name)
manifest=[]
for path in sorted(root.rglob('*')):
    if not path.is_file() or '__pycache__' in path.parts:continue
    relative=str(path.relative_to(root))
    if relative in ('manifest.sha256.json','restored-service.log','freeze.log'):continue
    before=path.stat();sha=digest(path);after=path.stat()
    if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns:
        raise RuntimeError('File changed while freezing: '+relative)
    manifest.append(dict(path=relative,bytes=after.st_size,sha256=sha))
(root/'manifest.sha256.json').write_text(json.dumps(dict(created=time.time(),files=manifest,
    omitted_mutable_files=['restored-service.log','freeze.log'],
    code_note='Final source and saved request/response evidence. Intermediate source bundles are also retained locally.'),indent=2)+'\n')
print('Manifest files:',len(manifest),flush=True)
