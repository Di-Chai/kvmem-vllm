#!/usr/bin/env python3
"""Answer-blind fixed-Q/K replay of four raw-Key farthest-first representatives.

This is a retrieval diagnostic, not an answer-generation benchmark. Existing
raw captures are memory mapped, never copied into the output. Targets are read
only after scores and native selections have been produced. No calibration Q,
target positions, answers or token text enter representative construction.
"""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

import analyze_block_graph as offline
from analyze_ruler_overlap_answers import ranks, span_coverage
from vllm_kvmem.config import KVMemConfig


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO/'artifacts/ruler-retrieval-diagnostic-20261023-r4'
DEFAULT_TARGETS = REPO/'artifacts/ruler-failures-20261022/analysis.json'


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    stat = Path(path).stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
        device=stat.st_dev, inode=stat.st_ino)


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def build_representatives(keys, counts, *, device='cpu', batch_pages=128):
    """Bound raw scratch; return CPU representatives and source token indices.

    ``keys`` is [L,sum(counts),KVH,D], with FP8 bytes or a floating dtype.
    Indices are offsets inside original blocks. Padding never participates.
    The FP16 cache exactly represents this experiment's original FP8 values.
    """
    from vllm_kvmem.query_key_fps import select_key_representatives
    require(len(keys.shape) == 4 and len(counts) > 0 and
        all(type(n) is int and 0 < n <= 32 for n in counts) and
        sum(counts) == keys.shape[1] and type(batch_pages) is int and
        0 < batch_pages <= 128, 'Invalid original Key partition or tile size')
    layers, _, heads, dim = keys.shape
    representatives = torch.empty((layers, len(counts), 4, heads, dim), dtype=torch.float16)
    indices = torch.empty((layers, len(counts), 4, heads), dtype=torch.long)
    begins = np.r_[0, np.cumsum(counts)]
    timings = dict(raw_read_decode_seconds=0., farthest_first_seconds=0.,
        representative_copy_seconds=0.)
    for layer in range(layers):
        for begin in range(0, len(counts), batch_pages):
            end = min(begin+batch_pages, len(counts))
            ns = counts[begin:end]
            sync(device); began = time.perf_counter()
            raw = torch.from_numpy(np.array(keys[layer, begins[begin]:begins[end]], copy=True))
            if raw.dtype == torch.uint8:
                raw = raw.view(torch.float8_e4m3fn)
            raw = raw.to(device=device, dtype=torch.float32)
            padded = torch.zeros((1, len(ns), 32, heads, dim), device=device)
            if all(n == 32 for n in ns):
                padded[0].copy_(raw.reshape(len(ns), 32, heads, dim))
            else:
                offset = 0
                for i, n in enumerate(ns):
                    padded[0, i, :n].copy_(raw[offset:offset+n])
                    offset += n
            sync(device); now = time.perf_counter()
            timings['raw_read_decode_seconds'] += now-began
            chosen, offsets = select_key_representatives(padded,
                torch.tensor(ns, dtype=torch.long, device=device))
            sync(device); finished = time.perf_counter()
            timings['farthest_first_seconds'] += finished-now
            representatives[layer, begin:end].copy_(chosen[0].to(device='cpu', dtype=torch.float16))
            indices[layer, begin:end].copy_(offsets[0].cpu())
            sync(device)
            timings['representative_copy_seconds'] += time.perf_counter()-finished
    timings['build_seconds'] = sum(timings.values())
    timings['representative_bytes'] = representatives.numel()*representatives.element_size()
    timings['index_bytes'] = indices.numel()*indices.element_size()
    return representatives, indices, timings


def score_capture(queries, keys, meta, *, device='cpu', batch_pages=128):
    """No target metadata or answers accepted by the scorer."""
    from vllm_kvmem.query_key_fps import key_fps_scores
    require(meta['dtype'] == 'float8_e4m3fn' and meta['raw_k_scale'] == 1.0,
        'This replay is qualified for the captured unit-scale raw FP8 content Keys only')
    require(list(keys.shape) == meta['key_shape'] and keys.dtype == np.uint8,
        'Raw capture shape or FP8 storage differs')
    representatives, indices, timings = build_representatives(keys, meta['counts'],
        device=device, batch_pages=batch_pages)
    sync(device); began = time.perf_counter()
    scores = key_fps_scores(queries.to(device), representatives, indices,
        torch.tensor(meta['score_mask'], dtype=torch.bool, device=device), query_half=True)
    scores = scores.cpu().numpy()
    sync(device)
    timings['score_seconds'] = time.perf_counter()-began
    require(np.isfinite(scores).all() and (scores >= 0).all(), 'Invalid FPS seed vector')
    require(np.all(scores[~np.asarray(meta['score_mask'], dtype=bool)] == 0),
        'FPS mass leaked outside native score mask')
    expected_mass = queries.shape[1] if any(meta['score_mask']) else 0
    require(abs(float(scores.sum())-expected_mass) < 1e-4*max(1, expected_mass),
        'Block-softmax/mean-head-layer/sum-Query normalization differs')
    timings.update(score_mass=float(scores.sum()), expected_score_mass=expected_mass)
    return scores, indices.to(torch.int8).numpy(), timings


def native_selections(profile, meta, variants):
    """Use the native masks, fixed retention and picker with graph alpha zero."""
    config = KVMemConfig(**{f.name:profile['kvmem'][f.name]
        for f in fields(KVMemConfig) if f.name in profile['kvmem']})
    require(config.block_tokens == 32, 'FPS experiment requires original 32-token blocks')
    starts, counts = meta['starts'], meta['counts']
    require(starts == list(range(0, meta['history'], 32)) and
        counts == [min(32, meta['history']-s) for s in starts],
        'Native diagnostic replay requires unchanged contiguous original history')
    memory, handles = offline._native_history(config, starts, counts)
    try:
        frame = memory._store.query_score_frame([handles[s] for s in starts],
            counts, meta['history'], [], meta['budget'])
        try:
            require(list(frame.score_mask) == meta['score_mask'] and
                list(frame.transport_mask) == meta['transport_mask'],
                'Captured score/transport domain differs from the native frame')
        finally:
            frame.close()
        selected = {}
        for method, scores in variants.items():
            require(len(scores) == len(starts) and np.isfinite(scores).all() and
                (np.asarray(scores) >= 0).all(), 'Invalid seed scores for native picker')
            require(np.all(np.asarray(scores)[~np.asarray(meta['score_mask'], dtype=bool)] == 0),
                'Seed scores contain mass outside the native score mask')
            slots = set(memory.preview_selection({handles[s].slot:float(v)
                for s, v in zip(starts, scores)}, (), action=False,
                prefill_scores=None, max_tokens=meta['budget']))
            selected[method] = [s for s in starts if handles[s].slot in slots]
            require(sum(n for s, n in zip(starts, counts) if s in set(selected[method]))
                <= meta['budget'], 'Native selection exceeded KV token budget')
        require(selected['captured_mean_k'] == meta['selected_starts'],
            'Captured MeanK native selection cannot be reproduced')
        return selected
    finally:
        require(memory.close(), 'Native diagnostic metadata did not release')


def evaluate_selection(scores, selected, meta, target):
    """Post-hoc evaluation only; target spans never affect scores or selection."""
    starts, counts = meta['starts'], meta['counts']
    selected = set(selected)
    ranking = ranks(scores, meta['transport_mask'])
    ranges = [(s, n) for s, n in zip(starts, counts) if s in selected]
    optional = [(s, float(v)) for s, v, keep in zip(starts, scores, meta['transport_mask'])
        if keep and s in selected]
    cutoff = min((v for _, v in optional), default=None)
    blocks = []
    for i, (s, n) in enumerate(zip(starts, counts)):
        value = max(0, min(s+n, target['token_span'][1])-max(s, target['token_span'][0]))
        key = max(0, min(s+n, target['key_token_span'][1])-max(s, target['key_token_span'][0]))
        if value or key:
            blocks.append(dict(start=s, count=n, score=float(scores[i]), rank=ranking[i],
                score_allowed=meta['score_mask'][i], transport_allowed=meta['transport_mask'][i],
                selected=s in selected, value_tokens=value, key_tokens=key,
                cutoff_ratio=float(scores[i])/cutoff if cutoff else None))
    return dict(value=span_coverage(ranges, target['token_span']),
        key=span_coverage(ranges, target['key_token_span']), target_blocks=blocks,
        optional_cutoff=cutoff, optional_selected_blocks=len(optional),
        selected_tokens=sum(n for _, n in ranges), selected_starts=sorted(selected),
        score_mass=float(np.sum(scores)),
        rank_definition='1 + count of strictly greater scores inside transport_mask; native picker decides ties',
        native_picker=True, graph_alpha=0, post_picker_neighbors_added=False)


def captured_references(source, case, meta):
    path = source/'analysis'/case/'Q_compressed_K_compressed.npz'
    with np.load(path) as previous:
        variants = {name:previous[key].mean(0).sum(0)
            for name, key in [('mean_k_replay', 'mean'), ('token_softmax_page_sum', 'token'),
                ('all_token_block_max', 'max')]}
    variants['captured_mean_k'] = np.asarray(meta['mean_scores'])
    return variants, dict(path=str(path.resolve()), sha256=sha(path),
        scope='Previously computed distributions from this exact compressed Q/K capture; no new answer generation')


def analyze_case(source, output, case, targets_path, *, device='cpu', batch_pages=128,
                 overlap_reference=None):
    directory = source/'compressed/cases'/case/'diagnostic'
    out = output/case
    out.mkdir(parents=True, exist_ok=False)
    files = {name:directory/name for name in ('queries.pt', 'raw_keys.npy', 'archive.json')}
    before = {name:identity(path) for name, path in files.items()}
    began = time.perf_counter()
    hashes = {name:sha(path) for name, path in files.items()}
    hash_seconds = time.perf_counter()-began
    meta = read(directory/'archive.json')
    queries = torch.load(directory/'queries.pt', weights_only=True)
    keys = np.load(directory/'raw_keys.npy', mmap_mode='r')
    fps, indices, timings = score_capture(queries, keys, meta, device=device, batch_pages=batch_pages)
    variants, reference = captured_references(source, case, meta)
    variants['key_fps_4'] = fps
    selected = native_selections(read(source/'compressed/profile.json'), meta, variants)
    np.save(out/'representative-token-offsets.npy', indices, allow_pickle=False)
    np.savez(out/'scores.npz', **variants)
    # Evaluation data is deliberately opened only after answer-blind artifacts.
    targets = read(targets_path)
    target = next(row for row in targets['rows'] if row['id'] == case)['references'][0]['spans'][0]
    evaluated = {name:evaluate_selection(values, selected[name], meta, target)
        for name, values in variants.items()}
    summary = dict(id=case, mode='key_fps_4', graph_alpha=0, device=device,
        value_span=target['token_span'], key_span=target['key_token_span'], variants=evaluated,
        timing=timings, hash_seconds=hash_seconds, source=str(directory.resolve()),
        source_sha256=hashes, source_identity=before, fixed_operand_reference=reference,
        source_stat_unchanged=all(identity(path) == before[name] for name, path in files.items()),
        generation_performed=False, representatives_per_block=4,
        selection_uses_targets=False, query_half=True, key_scale=1.0,
        score_coordinates='Original archived content Q/K, no extra RoPE; per-layer/head/Query representative max before block softmax',
        build_coordinates='Original raw Key magnitudes preserved; no Q, token text, targets or calibration input',
        timing_scope='Diagnostic cold mmap reads, construction and scoring; not model throughput or production archive/cache timing')
    require(summary['source_stat_unchanged'], 'Captured source files changed during replay')
    if overlap_reference:
        old = read(overlap_reference/'overlap_16_8/cases'/case/'result.json')
        summary['historical_overlap_reference'] = dict(root=str(overlap_reference.resolve()),
            full_value_coverage=old['value_covered'], official_case_score=old['official_case_score'],
            scope='Actual generated answer from another capture/run; not a fixed-Q/K counterfactual')
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--targets', type=Path, default=DEFAULT_TARGETS)
    case_group = parser.add_mutually_exclusive_group()
    case_group.add_argument('--case', action='append')
    case_group.add_argument('--cases', nargs='+')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--batch-pages', type=int, default=128)
    parser.add_argument('--overlap-reference', type=Path)
    args = parser.parse_args(argv)
    require(1 <= args.threads <= 8, 'Use between one and eight CPU threads')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = args.source.resolve()
    fixture = read(source/'fixture.json')
    known = [row['id'] for row in fixture['rows']]
    cases = args.case or args.cases or known
    require(len(set(cases)) == len(cases) and all(case in known for case in cases),
        'Cases must be unique original fixture cases')
    require(not args.output.exists(), 'Refusing to overwrite a prior diagnostic attempt')
    args.output.mkdir(parents=True)
    source_proof = dict(script_sha256=sha(Path(__file__)),
        scoring_module_sha256=sha(REPO/'src/vllm_kvmem/query_key_fps.py'))
    (args.output/'source-proof.json').write_text(json.dumps(source_proof, indent=2)+'\n')
    rows = []
    for case in cases:
        row = analyze_case(source, args.output, case, args.targets, device=args.device,
            batch_pages=args.batch_pages, overlap_reference=args.overlap_reference)
        rows.append(row)
        print(json.dumps(dict(id=case, value=row['variants']['key_fps_4']['value'],
            timing=row['timing'])), flush=True)
    require(source_proof == dict(script_sha256=sha(Path(__file__)),
        scoring_module_sha256=sha(REPO/'src/vllm_kvmem/query_key_fps.py')),
        'Diagnostic/scoring source changed during replay; retain this attempt for audit')
    result = dict(cases=rows, completed=len(rows), expected=len(cases), generation_performed=False,
        source=str(source), fixture_sha256=sha(source/'fixture.json'),
        **source_proof,
        summary={name:dict(full_value_coverage=sum(r['variants'][name]['value']['state'] == 'full' for r in rows),
            full_key_coverage=sum(r['variants'][name]['key']['state'] == 'full' for r in rows))
            for name in rows[0]['variants']},
        median_build_seconds=statistics.median(r['timing']['build_seconds'] for r in rows),
        median_score_seconds=statistics.median(r['timing']['score_seconds'] for r in rows),
        limits=['Failure-enriched twelve-case cohort, not overall RULER accuracy.',
            'Fixed captures were created before this scoring mode; future closed-loop Q/K may differ.',
            'Native picker coverage does not establish answer correctness.',
            'Source files were hashed before replay and stat-checked after; raw source files were not copied.'])
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    main()
