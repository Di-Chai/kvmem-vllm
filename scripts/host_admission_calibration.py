"""Empirical host launch allowance for one already-qualified FP8 replay.

This changes only the extra launch allowance. Complete history/index credits
and the live 8 GiB MemAvailable guard remain intact. Sampled system depletion
is an empirical envelope, not a process RSS bound or an allocation guarantee.
"""
import copy
import json
from pathlib import Path

GIB = 1024**3
SCHEMA = 'aligned_fp8_host_calibration_v1'
SAFETY_BYTES = 4 * GIB
MAX_REDUCTION_BYTES = 4 * GIB
MAX_SAMPLE_GAP_SECONDS = 5.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def derive(budget, admission, rows, timing):
    """Use the complete source run, including loading, warmup and drain."""
    require(budget['required_available_bytes'] == budget['archive_bytes'] +
            budget['mean_k_index_bytes'] + budget['extra_bytes'], 'Invalid original host budget')
    require(budget['extra_bytes'] == 32 * GIB and budget['minimum_free_bytes'] == 8 * GIB,
            'Calibration requires the original 32 GiB allowance and 8 GiB live guard')
    require(admission.get('passed') is True and len(admission.get('samples', [])) >= 3,
            'Source did not pass host admission')
    stable = admission['samples'][-3:]
    require(all(type(s['available_bytes']) is int and
                s['available_bytes'] >= budget['required_available_bytes'] and
                s['consecutive_qualifying_samples'] >= i + 1 for i, s in enumerate(stable)),
            'Source lacks three stable qualifying host samples')
    require(len(rows) >= 300, 'Insufficient all-phase memory samples')
    times = [r['at'] for r in rows]
    gaps = [b - a for a, b in zip(times, times[1:])]
    require(all(0 < gap <= MAX_SAMPLE_GAP_SECONDS for gap in gaps),
            'Memory samples are unordered or have a coverage gap')
    require(times[-1] - times[0] >= 600, 'Missing loading/warmup/drain coverage')
    require(0 <= rows[0]['wall_at'] - timing['started_wall'] <= 10 and
            0 <= timing['finished_wall'] - rows[-1]['wall_at'] <= 30 and
            stable[-1]['at'] <= rows[0]['wall_at'] <= stable[-1]['at'] + 30,
            'Memory samples do not cover the source server lifetime')
    available = [r['memory']['MemAvailable'] for r in rows]
    totals = [r['memory']['MemTotal'] for r in rows]
    require(len(set(totals)) == 1 and all(type(n) is int and 0 < n <= totals[0] for n in available),
            'Invalid system memory observations')
    minimum = min(available)
    require(minimum >= budget['minimum_free_bytes'], 'Source crossed the live memory guard')
    prelaunch = max(s['available_bytes'] for s in stable)
    depletion = max(0, prelaunch - minimum)
    # Keep an additional 4 GiB above the unchanged live guard and reduce the
    # old allowance by at most 4 GiB in this first calibration revision.
    spare = budget['required_available_bytes'] - depletion - budget['minimum_free_bytes'] - SAFETY_BYTES
    reduction = min(MAX_REDUCTION_BYTES, max(0, spare // GIB * GIB))
    require(reduction > 0, 'Measured peak does not justify lowering the launch allowance')
    extra = budget['extra_bytes'] - reduction
    required = budget['archive_bytes'] + budget['mean_k_index_bytes'] + extra
    projected = required - depletion
    require(extra >= budget['minimum_free_bytes'] + SAFETY_BYTES and
            projected >= budget['minimum_free_bytes'] + SAFETY_BYTES,
            'Calibration would violate the safety envelope')
    return dict(schema=SCHEMA, original_budget=copy.deepcopy(budget),
        extra_bytes=extra, required_available_bytes=required,
        reduction_bytes=reduction, minimum_free_bytes=budget['minimum_free_bytes'],
        additional_safety_bytes=SAFETY_BYTES, maximum_reduction_bytes=MAX_REDUCTION_BYTES,
        source_prelaunch_available_bytes=prelaunch, source_minimum_available_bytes=minimum,
        observed_maximum_depletion_bytes=depletion,
        projected_available_at_new_threshold_bytes=projected,
        observations=dict(count=len(rows), coverage_seconds=times[-1]-times[0],
                          maximum_gap_seconds=max(gaps)),
        scope='Same qualified model/device/replay only. Full256Ki history per user and index '
              'credits retained. All-phase sampled system depletion includes unrelated variation '
              'and can miss sub-second peaks; this is not an RSS bound or a guarantee for arbitrary '
              'workloads. The live8GiB guard and owned cleanup must remain enabled.')


def source_report(root, manifest, binding, sha):
    """Caller first validates the completed source's full qualification gates."""
    root = Path(root).resolve()
    names = ('rolling-candidate-v2-host-admission.json',
             'rolling-candidate-v2/gpu_samples.jsonl', 'rolling-candidate-v2/timing.json')
    digests = {name: sha(root/name) for name in names}
    admission = json.loads((root/names[0]).read_text())
    with (root/names[1]).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    timing = json.loads((root/names[2]).read_text())
    result = derive(manifest['budget'], admission, rows, timing)
    require(all(sha(root/name) == digest for name, digest in digests.items()),
            'Calibration observations changed while being read')
    result.update(source_run=str(root), source_binding=binding, observations_sha256=digests)
    return result


def calibrated_budget(original, report):
    require(report.get('schema') == SCHEMA and report['original_budget'] == original,
            'Calibration does not match the complete original resource budget')
    result = copy.deepcopy(original)
    result.update(extra_bytes=report['extra_bytes'],
                  required_available_bytes=report['required_available_bytes'],
                  scope='Full eight-user historical ceiling and complete index; calibrated '
                        'extra launch allowance, live8GiB guard; no output-length prediction')
    require(result['required_available_bytes'] == result['archive_bytes'] +
            result['mean_k_index_bytes'] + result['extra_bytes'] and
            result['minimum_free_bytes'] == report['minimum_free_bytes'] and
            original['extra_bytes'] - result['extra_bytes'] == report['reduction_bytes'] and
            0 < report['reduction_bytes'] <= MAX_REDUCTION_BYTES,
            'Invalid calibrated resource formula')
    return result


def observed_envelope(budget, admission, rows):
    require(admission.get('passed') is True and rows, 'Measured host envelope is missing')
    minimum = min(r['memory']['MemAvailable'] for r in rows)
    require(minimum >= budget['minimum_free_bytes'], 'Measured run crossed the unchanged live guard')
    prelaunch = [s['available_bytes'] for s in admission['samples'][-3:]]
    require(len(prelaunch) == 3 and min(prelaunch) >= budget['required_available_bytes'],
            'Measured run did not pass its calibrated launch rule')
    return dict(required_launch_available_bytes=budget['required_available_bytes'],
        extra_launch_bytes=budget['extra_bytes'], live_guard_bytes=budget['minimum_free_bytes'],
        actual_prelaunch_available_bytes=prelaunch,
        sampled_minimum_available_bytes=minimum,
        sampled_minimum_above_guard_bytes=minimum-budget['minimum_free_bytes'],
        observed_depletion_bytes=max(0, max(prelaunch)-minimum),
        sample_count=len(rows), sample_coverage_seconds=rows[-1]['at']-rows[0]['at'],
        live_guard_passed=True,
        scope='All server phases, including loading/warmup/drain. Sampled MemAvailable, '
              'not exclusive owned RSS or proof against sub-second allocation peaks.')
