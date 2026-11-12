"""Turn a chosen placement into a serving-profile overlay, environment and report."""
import json
import math

from .costs import STACK_ENV


def overlay(placement):
    """Keys to merge into a serving profile (top level and its ``kvmem`` section)."""
    kv = dict(session_gpu_checkpoint_bytes=0)      # no GPU state cache (user rule, 11-12)
    if placement.gpu_policy == 'none':
        kv.update(active_resident_max_fraction=0.01, active_resident_pressure_only=True,
                  active_resident_ttl_ms=0.0)
    elif placement.gpu_policy == 'cap':
        kv.update(active_resident_max_fraction=placement.residency_fraction,
                  active_resident_pressure_only=True, active_resident_ttl_ms=0.0)
    elif placement.gpu_policy == 'ttl':
        ttl = placement.gpu_ttl
        kv.update(active_resident_max_fraction=placement.residency_fraction,
                  active_resident_pressure_only=False,
                  active_resident_ttl_ms=(json.loads(json.dumps(ttl)) if isinstance(ttl, dict)
                                          else (0.0 if not math.isfinite(ttl) else ttl * 1000)))
    if placement.victim != 'fifo':
        kv['active_resident_victim'] = placement.victim          # C2
    if placement.archive == 'tiered' and math.isfinite(placement.cpu_ttl):
        # CPU -> SSD session tiering on the CPU archive (session_tiering.py).
        kv.update(cpu_session_ttl_ms=placement.cpu_ttl * 1000,
                  cpu_spill_dir='<ssd directory>', cpu_spill_max_bytes=0)   # filled by compile --ssd-dir
    return dict(max_num_seqs=placement.slots, kvmem=kv)


def environment(placement):
    env = STACK_ENV.get(placement.swap_stack)
    return dict(env) if env is not None else None


def merge(profile, patch):
    out = json.loads(json.dumps(profile))
    for key, value in patch.items():
        if key == 'kvmem':
            out.setdefault('kvmem', {}).update(value)
        else:
            out[key] = value
    return out


def report(workload, geometry, machine_info, chosen, ranked, *, extrapolated=(), best_any=None):
    lines = [f'# kvmem plan: {workload.name}, {workload.agents} agents', '']
    w = workload.describe()
    lines.append(f"Workload: window {w['window_tokens']} tokens, history {w['history_tokens']} "
                 f"(kept up to {w['max_history_tokens']}), {w['output_tokens']} output tokens/turn, "
                 f"tool wait mean {w['tool_wait_mean_s']} s ({w['tool_wait']}).")
    lines.append(f'Geometry: {geometry.pool_pages} pool pages of {geometry.page_bytes/2**20:.1f} MiB; '
                 f'window {geometry.window_pages(min(workload.window_tokens, geometry.select_budget or 10**9))} '
                 f'pages; {geometry.recurrent_pages_per_request} GDN pages per running request; '
                 f'{geometry.cpu_bytes_per_token} CPU bytes/token.')
    if machine_info:
        lines.append(f"Machine: RAM {machine_info.get('ram_bytes', 0)/2**30:.0f} GiB, memlock "
                     f"{machine_info.get('memlock_bytes')}, cores {machine_info.get('cores')}.")
        if machine_info.get('spill_read_gbps') is not None:
            lines.append(f"Spill device (measured): write {machine_info['spill_write_gbps']:.2f} GB/s, random read "
                         f"{machine_info['spill_read_gbps']:.2f} GB/s ({machine_info.get('spill_threads')} threads).")
    lines.append('')
    if chosen is None:
        lines.append('**No legal placement** under the memory and latency constraints.')
    else:
        r, m = chosen.result, chosen.memory
        lines += ['## Chosen', '',
                  '(Highest predicted throughput, unless a simpler placement is within 2 standard errors '
                  'or the model\'s ~3% validated error: then the simpler one.)', '',
                  f'`{chosen.placement.label()}`: **{r.output_tok_s:.0f} tok/s** predicted, turn latency '
                  f'p50 {r.turn_latency_p50:.1f} s / p90 {r.turn_latency_p90:.1f} s.',
                  f'Swap-ins per 5 min: {r.swaps}; evictions {r.evictions}; mean parked windows '
                  f'{r.mean_parked:.1f}.',
                  f'Memory: running {m.gpu_pages_running} pages, parked cap {m.gpu_pages_parked_cap} pages '
                  f'(~{m.parked_windows_fit} windows); CPU {m.cpu_total_bytes/2**30:.1f} GiB of '
                  f'{m.cpu_budget_bytes/2**30:.1f} GiB; SSD {m.ssd_bytes_needed/2**30:.1f} GiB; pinned '
                  f'{m.pinned_bytes/2**30:.1f} GiB.', '']
    if best_any is not None and best_any is not chosen:
        r = best_any.result
        lines += ['## Best once the runtime policies are built', '',
                  f'`{best_any.placement.label()}`: {r.output_tok_s:.0f} tok/s, p90 {r.turn_latency_p90:.1f} s '
                  f"(needs: {'; '.join(best_any.needs)}).", '']
    lines += ['## Top candidates', '', '| placement | tok/s (± s.e.) | p90 s | swaps (res/cpu/ssd) | deployable |',
              '|---|---|---|---|---|']
    for e in [e for e in ranked if e.result is not None][:12]:
        r = e.result
        se = f' ± {e.stderr:.0f}' if math.isfinite(e.stderr) else ''
        lines.append(f'| `{e.placement.label()}` | {r.output_tok_s:.0f}{se} | {r.turn_latency_p90:.1f} | '
                     f"{r.swaps['resident']}/{r.swaps['cpu']}/{r.swaps['ssd']} | "
                     f"{'yes' if e.deployable else 'needs ' + '; '.join(e.needs)} |")
    illegal = [e for e in ranked if not e.memory.legal]
    if illegal:
        lines += ['', '## Rejected by memory', '']
        for e in illegal[:6]:
            lines.append(f'- `{e.placement.label()}`: ' + '; '.join(e.memory.violations))
    if extrapolated:
        lines += ['', '## Caveats', ''] + [f'- {x}' for x in extrapolated]
    lines += ['', 'Model: engine-step simulation with costs fitted from captured runs '
              '(planner/costs.py); validated within ~3% on the 11-12 32-agent swap-stack runs, '
              '~6–13% elsewhere.']
    return '\n'.join(lines) + '\n'
