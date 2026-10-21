"""Component-only static FA2 readonly-load experiment; never a serving default.

First use --prepare-only to build distinct old/new modules and record exact .so
paths/SHA. Disassemble those libraries independently and fill a --sass-proof
JSON (schema below). Timing refuses to proceed unless loop metadata loads fall
without worsening registers, stack, or loop local loads. --qualify-only runs
byte qualification after this same gate but reports no performance samples.
No service setting or RuntimeOptions field activates this experiment.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time


MODES = ('static', 'readonly')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write('\n')


def validate_sass_proof(proof, identities):
    """Require reviewed source/resource evidence before any component timing.

    Proof keys: schema=1, fp_body_unchanged=true, reference/candidate mappings
    with library_sha256, overlay, registers, stack_bytes, loop_metadata_loads,
    loop_local_loads, sass_file, sass_sha256. The disassembly remains attached
    evidence; this function does not pretend a text count proves correct loops.
    """
    if proof.get('schema') != 1 or proof.get('fp_body_unchanged') is not True:
        raise ValueError('SASS proof needs explicit unchanged arithmetic review')
    for side in ('reference', 'candidate'):
        evidence, actual = proof.get(side, {}), identities[side]
        if (evidence.get('library_sha256') != actual['library_sha256'] or
                evidence.get('overlay') != actual['overlay']):
            raise ValueError('SASS proof does not bind the actual compiled library')
        path = Path(evidence.get('sass_file', ''))
        if not path.is_file() or sha(path) != evidence.get('sass_sha256'):
            raise ValueError('Missing or changed SASS evidence')
        for field in ('registers', 'stack_bytes', 'loop_metadata_loads', 'loop_local_loads'):
            if type(evidence.get(field)) is not int or evidence[field] < 0:
                raise ValueError('Invalid SASS resource count: ' + field)
    old, new = proof['reference'], proof['candidate']
    if new['loop_metadata_loads'] >= old['loop_metadata_loads']:
        raise ValueError('No proved reduction in loop metadata loads; stop before timing')
    if any(new[key] > old[key] for key in ('registers', 'stack_bytes', 'loop_local_loads')):
        raise ValueError('Readonly candidate worsens register/spill resources; stop before timing')
    return True


def build_modules(output):
    import torch
    from flashinfer.jit.core import jit_spec_registry
    from vllm_kvmem.native_attention_build import prefill_module, source_overlay
    geometry = (torch.bfloat16, torch.float8_e4m3fn, torch.bfloat16, torch.int32, 256)
    reference_overlay = source_overlay(None, prefill_static_aligned=True)
    candidate_overlay = source_overlay(None, prefill_static_aligned=True, prefill_static_readonly=True)
    if reference_overlay.name == candidate_overlay.name:
        raise RuntimeError('Readonly and old static share a source fingerprint')
    reference = prefill_module(output.parent / 'build', *geometry, static_aligned=True)
    candidate = prefill_module(output.parent / 'build', *geometry, static_aligned=True, static_readonly=True)
    specs = jit_spec_registry.get_all_specs()
    found = [spec for name, spec in specs.items()
             if reference_overlay.name in name and name.endswith('_static_aligned16')]
    if len(found) != 1:
        raise RuntimeError('Cannot uniquely identify original static compiled library')
    old_library = found[0].get_library_path()
    identities = {}
    for side, overlay, library in [('reference', reference_overlay, old_library),
                                   ('candidate', candidate_overlay, Path(candidate.library_path))]:
        identities[side] = dict(overlay=overlay.name, library=str(library), library_sha256=sha(library),
            native_header_sha256=sha(overlay / 'native_prefill_stats.cuh'))
    return geometry, reference, candidate, identities


def setup(length, output, *, qlen, max_rows, modules):
    import benchmark_native_prefill_stats_floor as floor
    geometry, reference, candidate, identities = modules
    run, arms, audit, metadata = floor.setup(length, output, qlen=qlen, max_rows=max_rows)
    # Both fixtures retain their original fixed buffers/Action paths. Replace
    # only the prewarmed static module selected by the normal host qualification.
    for enabled, selected, side in ((False, reference, 'reference'), (True, candidate, 'candidate')):
        action, bank, _, _ = arms[enabled]
        bank.static_prefill_enabled = True
        bank.static_prefill_unavailable = None
        bank.static_prefill_source_fingerprint = identities[side]['overlay']
        bank.prefill_modules[geometry + ('static_aligned16',)] = selected
    def execute(mode, **kwargs):
        translated = {'static': 'generic', 'readonly': 'static_aligned', 'native': 'native'}[mode]
        return run(translated, **kwargs)
    return execute, run, arms, audit, metadata


def kernel_profile(run, mode):
    import torch
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
        run(mode)
        torch.cuda.synchronize()
    events = [event for event in profile.events() if str(event.device_type).endswith('CUDA')]
    attention = [event for event in events if 'BatchPrefillWithPagedKVCacheKernel' in event.name]
    expected = {'static': 'KVMemStaticAlignedAttention', 'readonly': 'KVMemStaticReadonlyAttention'}[mode]
    if len(attention) != 3 or any(expected not in event.name for event in attention):
        raise RuntimeError('Actual FA2 specialization differs from the component arm')
    return dict(fa2_kernel_ms=sum(event.device_time_total for event in attention) / 1000,
        all_gpu_kernels_ms=sum(event.device_time_total for event in events) / 1000,
        fa2_kernels=[dict(name=event.name, duration_us=event.device_time_total) for event in attention])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--qualify-only', action='store_true')
    parser.add_argument('--sass-proof', type=Path)
    parser.add_argument('--expected-static-overlay', default='d7b326c5139bf4d9')
    parser.add_argument('--lengths', type=int, nargs='+', default=[16384, 32768, 49152])
    parser.add_argument('--query-tokens', type=int, nargs='+', default=[2048, 8192, 14336])
    parser.add_argument('--repeats', type=int, default=12)
    parser.add_argument('--profile-repeats', type=int, default=3)
    parser.add_argument('--max-rows', type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output artifact')
    if args.prepare_only and args.qualify_only:
        raise ValueError('Choose preparation or qualification')
    if min(args.repeats, args.profile_repeats, args.max_rows) <= 0:
        raise ValueError('Positive repetition/capacity required')
    if any(q <= 0 or q % 32 for q in args.query_tokens) or any(n <= 0 for n in args.lengths):
        raise ValueError('Invalid component geometry')
    if not args.prepare_only and args.sass_proof is None:
        raise ValueError('Reviewed SASS improvement is required before GPU qualification/timing')
    modules = build_modules(args.output)
    identities = modules[3]
    if identities['reference']['overlay'] != args.expected_static_overlay:
        raise RuntimeError('Historical static source/hash changed')
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / 'scripts/benchmark_native_prefill_stats_floor.py']
    sources += [root / 'src/vllm_kvmem' / name for name in (
        'native_attention_build.py', 'native_prefill_readonly.py', 'native_attention.py',
        'native_stats.cuh', 'native_prefill_stats.cuh')]
    evidence = dict(schema=1, scope='Component-only readonly metadata; fixed sampled rows and unchanged FP body',
        source_sha256={str(path.relative_to(root)):sha(path) for path in sources}, compiled=identities,
        mode='prepare' if args.prepare_only else 'qualification' if args.qualify_only else 'paired_component',
        default_service_unchanged=True, results=[])
    if args.prepare_only:
        evidence['sass_plan'] = dict(tool='cuobjdump --dump-resource-usage / --dump-sass',
            cta_tile_q=64, old_reference=dict(registers=255, stack_bytes=32, loop_metadata_loads=4, loop_local_loads=3),
            requirement='Inspect the actual main KV loop; same arithmetic and fewer metadata loads with no register/spill regression. Bind both SASS files by SHA in --sass-proof. Otherwise stop.')
        write_new(args.output, evidence)
        return
    proof = json.loads(args.sass_proof.read_text())
    validate_sass_proof(proof, identities)
    evidence['sass_proof'] = dict(path=str(args.sass_proof.resolve()), sha256=sha(args.sass_proof), proof=proof)
    import torch
    import benchmark_native_prefill_stats_floor as floor
    for qlen in args.query_tokens:
        for length in args.lengths:
            if qlen > length:
                continue
            execute, original_run, arms, audit, item = setup(length, args.output,
                qlen=qlen, max_rows=args.max_rows, modules=modules)
            before_checks = modules[2].checked_dispatches
            # Existing floor qualifier compares each layer O/native LSE, full
            # capacity piece-LSE including sentinels, and final page scores.
            floor.qualify(original_run, audit)
            for _ in range(3):
                for mode in MODES:
                    del_result = execute(mode)
                    del del_result
            torch.cuda.synchronize()
            item['bitwise_outputs_row_lse_piece_lse_scores'] = True
            item['samples'] = []
            if not args.qualify_only:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                for repetition in range(args.repeats):
                    for mode in MODES if repetition % 2 == 0 else MODES[::-1]:
                        torch.cuda.synchronize()
                        began = time.perf_counter(); start.record()
                        result = execute(mode); end.record(); end.synchronize()
                        item['samples'].append(dict(repetition=repetition, mode=mode,
                            wall_ms=1000 * (time.perf_counter() - began), cuda_interval_ms=start.elapsed_time(end)))
                        del result
                item['summary'] = {mode:{field:statistics.mean(row[field] for row in item['samples'] if row['mode']==mode)
                    for field in ('wall_ms', 'cuda_interval_ms')} for mode in MODES}
                # Independent profiling is excluded from wall/event samples.
                item['kernel_profiles'] = {mode:[kernel_profile(execute, mode) for _ in range(args.profile_repeats)] for mode in MODES}
            item['checked_readonly_dispatches'] = modules[2].checked_dispatches - before_checks
            if item['checked_readonly_dispatches'] <= 0:
                raise RuntimeError('Readonly compiled path was never exercised')
            item['dispatch_audit'] = audit()
            item['stats'] = {}
            for enabled, (_, bank, _, _) in arms.items():
                side = 'readonly' if enabled else 'static'
                if not bank.static_prefill_counts.get('static_dispatch_calls') or any(
                        key.startswith('generic_dispatch:') for key in bank.static_prefill_counts):
                    raise RuntimeError('A component arm did not use static specialization')
                item['stats'][side] = dict(counts=dict(bank.static_prefill_counts), allocated_bytes=bank.allocated_bytes,
                    source_fingerprint=bank.static_prefill_source_fingerprint)
            assert item['stats']['static']['allocated_bytes'] == item['stats']['readonly']['allocated_bytes']
            evidence['results'].append(item)
            del execute, original_run, arms, audit, item
            gc.collect()
            torch.cuda.synchronize()
    for path in sources:
        if sha(path) != evidence['source_sha256'][str(path.relative_to(root))]:
            raise RuntimeError('Source changed during qualification/component')
    evidence['source_unchanged'] = True
    write_new(args.output, evidence)


if __name__ == '__main__':
    main()
