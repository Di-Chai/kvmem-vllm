"""Exact memory budget of a placement: GPU pool pages, CPU RAM, SSD and pinned bytes.

These are hard constraints. The planner rejects any candidate that violates one
and reports which constraint bound. GPU memory is the fixed KV pool only; no
placement may add buffers outside it.
"""
from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class Placement:
    slots: int                          # running requests (max_num_seqs)
    gpu_policy: str = 'cap'             # 'cap' (residency fraction, FIFO), 'ttl', 'hazard', 'none'
    residency_fraction: float = 0.5     # pool share parked windows may keep (cap/ttl/hazard)
    gpu_ttl: float = float('inf')       # seconds a parked window stays on GPU ('ttl')
    victim: str = 'fifo'                # 'fifo' or 'expected_wait'
    archive: str = 'cpu'                # 'cpu' or 'tiered' (CPU + SSD)
    cpu_ttl: float = float('inf')       # seconds before an idle session moves to SSD ('tiered')
    pinned_archive: bool = False        # page-locked archive (not built; modelled only)
    swap_stack: str = 'direct_prepared_exempt'   # swap-in implementation (see costs.SWAP_STACKS)
    checkpoints_per_session: int = 2    # GDN states kept on CPU per parked session (P and M)

    def label(self):
        parts = [f'S{self.slots}', self.gpu_policy]
        if self.gpu_policy in ('cap', 'ttl', 'hazard'):
            parts.append(f'R{self.residency_fraction:g}')
        if self.gpu_policy == 'ttl':
            parts.append(f'ttl{self.gpu_ttl:g}s')
        if self.victim != 'fifo':
            parts.append(self.victim)
        if self.archive == 'tiered':
            parts.append(f'ssd@{self.cpu_ttl:g}s')
        if self.pinned_archive:
            parts.append('pinned')
        parts.append(self.swap_stack)
        return '/'.join(parts)


@dataclass(frozen=True)
class Machine:
    ram_bytes: int
    ram_reserved_bytes: int             # OS, vLLM process, model staging, other tenants
    memlock_bytes: float                # RLIMIT_MEMLOCK / cgroup (inf when unlimited)
    pinned_fraction_limit: float = 0.5  # never page-lock more than this share of RAM
    ssd_bytes: int = 0
    cpu_archive_budget_bytes: int = 0   # serving profile's CPU archive budget (0: RAM-derived)


@dataclass
class MemoryPlan:
    gpu_pages_total: int
    gpu_pages_running: int
    gpu_pages_parked_cap: int
    parked_windows_fit: int
    cpu_archive_bytes: int
    cpu_checkpoint_bytes: int
    cpu_total_bytes: int
    cpu_budget_bytes: int
    ssd_bytes_needed: int
    pinned_bytes: int
    violations: list = field(default_factory=list)

    @property
    def legal(self):
        return not self.violations

    def to_dict(self):
        d = asdict(self)
        d['legal'] = self.legal
        return d


def plan_memory(geometry, workload, placement, machine):
    v = []
    window = min(workload.window_tokens, geometry.select_budget or workload.window_tokens)
    running = placement.slots * geometry.running_pages(window)
    total = geometry.pool_pages
    if running > total:
        v.append(f'GPU pool: {placement.slots} running requests need {running} pages, pool has {total}')
    parked_cap = 0 if placement.gpu_policy == 'none' else int(total * placement.residency_fraction)
    parked_cap = max(0, min(parked_cap, total - running))
    fit = parked_cap // max(1, geometry.window_pages(window))

    history = min(workload.max_history_tokens,
                  workload.history_tokens + 40 * (workload.append_median + workload.output_tokens))
    archive = workload.agents * history * geometry.cpu_bytes_per_token
    checkpoints = workload.agents * placement.checkpoints_per_session * geometry.recurrent_state_bytes
    budget = machine.cpu_archive_budget_bytes or max(0, machine.ram_bytes - machine.ram_reserved_bytes)
    cpu_needed = archive + checkpoints
    ssd = 0
    if placement.archive == 'tiered':
        # Sessions beyond the CPU budget live on SSD; checkpoints stay in RAM.
        ssd = max(0, cpu_needed - budget)
        if ssd > machine.ssd_bytes:
            v.append(f'SSD: needs {ssd/2**30:.1f} GiB, has {machine.ssd_bytes/2**30:.1f} GiB')
        cpu_total = min(cpu_needed, budget)
    else:
        cpu_total = cpu_needed
        if cpu_needed > budget:
            v.append(f'CPU RAM: archive + checkpoints need {cpu_needed/2**30:.1f} GiB, '
                     f'budget {budget/2**30:.1f} GiB (use tiered SSD, a history limit or fewer agents)')
    pinned = checkpoints + (min(archive, budget) if placement.pinned_archive else 0)
    pin_limit = min(machine.memlock_bytes, machine.pinned_fraction_limit * machine.ram_bytes)
    if placement.pinned_archive and pinned > pin_limit:
        v.append(f'Pinned: {pinned/2**30:.1f} GiB exceeds limit {pin_limit/2**30:.1f} GiB')
    return MemoryPlan(gpu_pages_total=total, gpu_pages_running=running, gpu_pages_parked_cap=parked_cap,
                      parked_windows_fit=fit, cpu_archive_bytes=archive, cpu_checkpoint_bytes=checkpoints,
                      cpu_total_bytes=cpu_total, cpu_budget_bytes=budget, ssd_bytes_needed=ssd,
                      pinned_bytes=pinned, violations=v)
