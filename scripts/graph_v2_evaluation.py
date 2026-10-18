"""Freeze original evaluators and public reference judgments without credentials."""
import json
from pathlib import Path
import shutil
import subprocess

from graph_v2_protocol import require, sha

EVALUATORS = Path('/home/chaidi/qw3/benchmark/qw3_memory_comparison_20261004/frozen')
OFFICIAL = Path('/home/chaidi/MemoryAgentBench-official')
ENTITY_MAP = Path('/home/chaidi/kvmem_eval/KVMem_Motivation/data/raw/MemoryAgentBench/entity2id.json')
EVAL_PYTHON = Path('/home/chaidi/kvmem_eval/KVMem_Motivation/.venv/bin/python')
PINNED_COMMIT = '455306dcabc3842526eb83cd4e225e5d486c5c5d'
EVALUATOR_FILES = ('score_memoryagentbench_official_local.py', 'judge_memoryagentbench_special.py',
    'historical_lme_score.py', 'historical_lme_evaluate_qa.py', 'historical_lme_calibration.json')
PACKAGES = ('numpy', 'nltk', 'tiktoken', 'rouge-score', 'editdistance', 'requests')


def interpreter_identity(python):
    # Query package metadata only. No scorer, credential loader, or API client runs.
    program = ('import importlib.metadata as m,json,sys;print(json.dumps(dict('
        'version=sys.version,packages={name:m.version(name) for name in '+repr(PACKAGES)+'})))')
    return json.loads(subprocess.check_output([str(python), '-B', '-c', program], text=True))


def copy_file(source, destination):
    require(source.is_file(), 'Missing frozen evaluator/reference dependency: '+str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def freeze_reference_evidence(source, destination, suite, rows):
    """Exact answer-bound cache and original comparison; no live judge state."""
    names = ['comparison.json']
    for arm in ('attention', 'query'):
        prefix = 'results/'+arm+'/'
        if suite == 'memoryagentbench':
            names += [prefix+'memoryagentbench/'+name for name in
                ('judge-answer-manifest.json', 'special_judgments.jsonl')]
        else:
            names += [prefix+'judge_deepseek_v41_flash/'+name for name in
                ('identity.json', 'calibration.json')]
            names.append(prefix+'longmemeval_judgments.jsonl')
            for row in rows:
                names += [prefix+'cases/longmemeval_s/'+row['id']+'/result.json',
                    prefix+'judge_deepseek_v41_flash/requests/'+row['id']+'.request.json']
    for name in names:
        copy_file(source/name, destination/name)
    return {name:sha(destination/name) for name in names}


def freeze_evaluation(root):
    target = root/'evaluation'
    target.mkdir()
    for name in EVALUATOR_FILES:
        copy_file(EVALUATORS/name, target/'evaluators'/name)
    head = subprocess.check_output(['git','-C',str(OFFICIAL),'rev-parse','HEAD'],text=True).strip()
    require(head == PINNED_COMMIT, 'Official evaluator repository differs from pinned commit')
    # A real independent local clone keeps the unchanged evaluators' git HEAD
    # checks valid. No hardlinks, alternates, remote access, or source mutation.
    official = target/'official'
    subprocess.run(['git','clone','--quiet','--no-hardlinks','--no-checkout',str(OFFICIAL),str(official)],check=True)
    subprocess.run(['git','-C',str(official),'checkout','--quiet','--detach',PINNED_COMMIT],check=True)
    subprocess.run(['git','-C',str(official),'remote','remove','origin'],check=True)
    copy_file(ENTITY_MAP, official/'processed_data/Recsys_Redial/entity2id.json')
    hashes = {str(path.relative_to(target)):sha(path) for path in target.rglob('*') if path.is_file()}
    manifest = dict(schema='kvmem-graph-v2-evaluation-1', official_commit=PINNED_COMMIT,
        evaluator_files=EVALUATOR_FILES, files_sha256=hashes,
        python=str(EVAL_PYTHON), interpreter_identity=interpreter_identity(EVAL_PYTHON),
        protocol='Unmodified original official/local/DeepSeek evaluators and exact answer-bound reuse only')
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return sha(target/'manifest.json')


def validate_evaluation(root, expected):
    target = root/'evaluation'
    require(expected and sha(target/'manifest.json') == expected, 'Frozen evaluation manifest changed or missing')
    manifest = json.loads((target/'manifest.json').read_text())
    require(manifest.get('schema') == 'kvmem-graph-v2-evaluation-1', 'Unknown evaluation bundle')
    for name, digest in manifest['files_sha256'].items():
        require(sha(target/name) == digest, 'Frozen evaluator dependency changed: '+name)
    require(interpreter_identity(Path(manifest['python'])) == manifest['interpreter_identity'],
        'Frozen evaluator interpreter/package versions changed')
    return manifest


def validate_answers(dataset, fixture, *, allow_partial=False):
    """Never turn failed/interrupted answers into silently omitted scores."""
    manifest = json.loads((dataset/'manifest.json').read_text())
    require(manifest['rows'] == fixture['rows'], 'Answer manifest differs from the frozen source cohort')
    expected = {row['id']:row for row in fixture['rows']}
    present = {path.parent.name:path for path in (dataset/'cases').glob('*/result.json')}
    require(not set(present)-set(expected), 'Unexpected answer outside the frozen cohort')
    completed, missing = [], []
    for ident, source in expected.items():
        if ident not in present:
            require(not (dataset/'cases'/ident/'attempt.json').exists(), 'Interrupted answer must be investigated: '+ident)
            missing.append(ident)
            continue
        record = json.loads(present[ident].read_text())
        require(record.get('status') == 'completed', 'Failed answer must be investigated: '+ident)
        require(record.get('id') == ident and record.get('suite') == fixture['suite'], 'Answer identity changed')
        require(record.get('token_sha256') == source['token_sha256'] and
            record.get('prompt_sha256') == source['prompt_sha256'] and
            record.get('query_sha256') == source['query_sha256'] and
            record.get('prompt_tokens') == source['input_tokens'], 'Answer source binding changed: '+ident)
        completed.append(record)
    require(allow_partial or not missing, 'Incomplete cohort; use --allow-partial only for explicit diagnostic analysis')
    require(completed, 'No completed answers to analyze')
    if fixture['suite'] == 'memoryagentbench':
        groups = {}
        for row in completed:
            groups.setdefault(row['id'].rsplit('_q',1)[0], []).append(row)
        for group in groups.values():
            require(len({row['prefix_sha256'] for row in group}) == 1 and
                len({row['attention']['prefill_attention_samples'] for row in group}) == 1 and
                all(row['attention']['action_history_pages'] == 0 and
                    row['attention']['attention_source'] == 'prefill' for row in group) and
                sum(not row['prefix_reused'] for row in group) == 1,
                'Independent frozen-P questions inherited different prefix/output statistics')
    return dict(expected=len(expected), completed=len(completed), missing=missing,
        complete=not missing, diagnostic_partial=bool(missing), failures_silently_skipped=False)
