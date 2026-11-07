# System One active-KV observation branch

`fork_cobatch` observes a frozen, accepted KVMem history while the foreground
continues generating. Observer question chunks and foreground target tokens
execute in the same native forward batch. This is a bounded System One label
read, not a general second autoregressive completion API.

## Enable and call

Start the existing KVMem server with `--systemone-fork`. The qualified profile
uses eager execution, synchronous refresh/restore, FP8 KV,
one GPU, and either no speculation or MTP. Prefix caching must be disabled.
`--systemone-branches 1|2|4` reserves independent observer slots; set
`max_num_seqs` to branches plus one. Only one foreground owner runs at a time.
`--systemone-quantum 16` limits tokens per observer per forward (4–64).
`--systemone-token-budget` limits total observer tokens per forward and defaults
to branches times quantum. Allow at least four tokens per reserved branch.
The latency experiment uses quantum=64 and the default total budget.

While a foreground completion with `session_id` and `cache_salt` is active,
send the following to `POST /v1/systemone`, using the same served model and salt:

```json
{
  "model": "systemone-fork-4090",
  "state": "",
  "cache_salt": "example-session",
  "kvmem_session": {
    "session_id": "example-session",
    "mode": "fork_cobatch"
  },
  "questions": {
    "damaged": {
      "type": "noul",
      "instructions": "Was the parcel damaged?"
    }
  }
}
```

Optional `generation` and `history_end` reject a stale owner or boundary. The
snapshot is taken at an eligible accepted decode boundary, not necessarily at
HTTP arrival. All questions in one call use that same snapshot. Put additional
instructions in the questions; top-level `state` must be empty and top-level
instructions absent. The existing native renderer and label strategies remain
responsible for answer semantics. The limit is eight questions, each at most
512 rendered tokens before the bounded turn separator. Omit `mode` to retain
the existing serialized `active_branch` behavior.

Concurrent requests execute in independent branches up to the reserved capacity.
Ungrouped requests freeze their history when they reach an eligible boundary,
so they may observe different H values. For an atomic two- or four-way fork,
add the same group to each request's `kvmem_session`:

```json
{
  "session_id": "example-session",
  "mode": "fork_cobatch",
  "fork_group": {"id": "unique-observation-round", "size": 4}
}
```

Submit exactly `size` concurrent requests with matching session, cache salt,
group ID, and size. They wait for all members and enough private slots, then
fork at the same accepted H before any model forward. Each branch has its own
question list, state, suffix, output coordinates, and cancellation lifetime.
Use a fresh group ID per round. A stale member rejects the group before
allocation; an incomplete group times out without GPU ownership. Canceling a
waiting member rejects its waiting peers. Canceling an admitted member leaves
its peers running.

Up to eight questions in one request share one branch and run in order.
Separate grouped requests execute their question chunks in the same target batch.

## Ownership and execution

1. The scheduler reserves private attention and recurrent storage before
   admitting a foreground request, including enough space for worst-case
   observer copy-on-write. The private native request slot is also reserved.
2. At the accepted boundary H, record the compressed active length W and
   logical-position offset H−W. Pin the foreground's complete attention pages
   by reference count. Copy the partial tail into a private page. Canonical
   accepted conv/SSM state and acceptance counters are independent of the
   foreground and of speculative rows.
3. Register private native request/model/encoder state. Each target forward
   appends one question chunk per observer to the foreground batch, with separate
   page tables and positions. Reset the private recurrent state before each
   question so all questions see the same H.
4. Before public sampling, MTP, and archive publication, project hidden states,
   metadata, and captured Q/K/V back to the public rows. Private question
   tokens never enter public history or the CPU archive.
5. A foreground refresh can overwrite active pages even when referenced by
   the observer. Before executing that refresh, copy the shared reader prefix
   into already reserved private pages, update the private table, fence the
   copy, and release pins. The observer skips that public refresh step and
   resumes afterward; it retains its original H and W.
6. Completion, cancellation, timeout, or foreground retirement removes private
   native registrations and releases references exactly once. Scheduler notices
   are consumed before native parent retirement can free its pages. The fixed
   startup reservation remains available for the next observation.

There is no parent rollback in ordinary `fork_cobatch` execution. Fork setup,
copies, and larger forward batches still consume GPU/CPU time. Foreground
streaming can continue during observation; zero additional token latency is
not promised. A question suffix is additional computation even when the
preceding history is shared.

## Verification and evidence

`verify_session_restore=true` enables byte audits. These include an identical
batch-layout counterfactual: change one branch's input content at a time,
then verify foreground and all other observers' hidden states, written attention
KV, and recurrent state remain byte-identical. Audit snapshots live on CPU to
bound additional GPU memory. The audit restores the actual result before public sampling;
its extra forward and rollback are deliberately absent from latency runs.

`--verify-systemone-refresh` additionally exercises the real scheduler refresh
path during a live shared lease. It requires the byte-audit profile. This is
needed for the experiment because ordinary request planning can reserve enough
window budget to avoid any in-response refresh. Child prefix and recurrent
bytes are checked across that refresh. Never enable this flag for latency
measurement or ordinary service operation. A no-fork control does not receive
this injected refresh, so that comparison cannot establish equal full-generation
trajectories under the same refresh policy.

CPU ownership/projection tests are in `tests/test_systemone_fork.py` and
`tests/test_systemone_multifork.py`. The multi-branch GPU probes are in
`experiments/agent_execution_observer/fork_multibranch_probe.py` and
`fork_multibranch_lifecycle.py`; they require the repository `scripts` directory on
`PYTHONPATH`. They record raw foreground SSE, native page/state audit events,
HTTP results, and release evidence. The 4090 experiment, configurations,
checksums, and measured limitations are under
`artifacts/systemone-multifork-4090-20261105-r1/`. The earlier single-branch
qualification is in `artifacts/systemone-fork-4090-20261105-r1/`.

## Current limits

The validated deployment is the Qwen 27B GPTQ Int4 hybrid model with FP8 KV,
eager execution, and MTP=1 on one RTX 4090. CUDA graphs, DFlash, asynchronous
refresh/restore, multiple foreground owners, distributed execution, LoRA,
multimodal input, and prompt embeddings are outside this qualification.
The RPC observation deadline is 30 seconds; a stalled engine can delay its
delivery and cleanup beyond that wall-clock time.

Changing batch shape is not guaranteed to preserve foreground token-for-token
reproducibility. The test records that comparison separately from direct
memory-isolation checks. This experiment verifies execution and ownership;
it does not establish trajectory-segmentation accuracy.

## RTX 5090 numerical diagnosis (2026-11-06)

The current source snapshot was also tested with vLLM 0.32.0, the Qwen 27B
ModelOpt NVFP4/FP8 mixed model, FP8 E4M3 KV, eager MTP=1, and K/G=2048/2048.
With forced refresh disabled, the 0/1/2/0 observer sweep generated 512 tokens
per arm. The two controls were identical; both fork arms first differed at
output index 73. Every arm had the same sole refresh at history 4032.

An external diagnostic restored identical pre-forward KV and recurrent state,
then compared the real mixed batch with a native foreground-only forward.
At history 5111, two foreground tokens became batches of 18 or 34 tokens.
The foreground-only replay chose token 92217, while the one- and two-observer
batches chose 64700 and 29896, exactly reproducing the first observed divergence.
Maximum absolute logit differences over the two public rows were 2.1484375 and
1.6328125; the foreground-only winner's margin was just 0.125.

Before any cache access, the first layer's normalization output was identical,
but its FP8 QKVZ and unquantized BA projections already differed by up to
0.00390625 and 0.03125. Thus numerical dependence on batch shape is directly
observed; the downstream contribution of each individual kernel is not isolated.
Six same-layout replays reproduced all hidden, recurrent, and written KV bytes;
historical prefix bytes stayed unchanged. Changing one observer's content also
left the foreground and other observer byte-identical. All four instrumented
160-token outputs matched the corresponding uninstrumented prefixes.

This explains the reproduced first divergence without evidence of cross-owner
KV/state corruption. It does not guarantee task quality or token invariance for
other inputs. Later KV and recurrent values can legitimately diverge as changed
numerical outputs feed back into generation. The earlier forced refresh at
history 5113 occurred after this first differing output and additionally changes
the later context policy.

Evidence and the isolated diagnostic scripts are in
`artifacts/systemone-fork-diagnosis-20261106-r1/summary.json` and
`numeric-r4/server/capture/numeric_probe.jsonl` beneath that directory.
No diagnostic hook was added to ordinary serving code. The test service was
stopped, with no GPU processes left and port 18996 closed.
