"""kvmem plan: choose a KVMem placement for a workload on a given machine.

The planner is offline and never touches a running server. It combines

* geometry  - bytes per token, GPU pages, GDN state sizes (model + cache layout),
* hardware  - VRAM/KV pool, RAM, pinned limits, measured copy bandwidths,
* workload  - agents, active window, history, turn shape, tool-wait distribution,
* costs     - step and swap costs fitted from captured runs,

into an exact memory budget (hard constraints) and an event simulation that
predicts throughput and turn latency. A grid search over placements (slots,
GPU TTL / residency, CPU TTL to SSD, archive backend, swap-in flags) picks the
best legal configuration and emits a profile overlay plus a short report.

Usage: python -m vllm_kvmem.apps.planner {probe,calibrate,compile,validate} ...
"""
