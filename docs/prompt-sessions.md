# Experimental exact prompt sessions

This path has passed short and 134K compressed actual-model HTTP/GPU token-equality
gates, plus a four-session lifecycle/cancellation gate. Concurrent numerical quality
and agent performance are still pending. It is disabled by default and is separate from native streaming-input
sessions. Target Q8_0, target FP8 e4m3, DFlash2 BF16 sliding KV and K128K/G64K are
unchanged.

Enable `session_capacity` and `session_checkpoint_bytes` together in the public
connector's `kv_connector_extra_config.kvmem` object. The existing bounded runner,
scheduler, continuous authority capture and `async_archive_retirement` must be
enabled. The qualified-recipe backend manager exposes the two extra options:

```sh
.venv/bin/python scripts/serve_agent_backend.py \
  --recipe artifacts/engine_long_c4_dflash_tensorcore_01/results.json \
  --output artifacts/session_backend_02 \
  --session-capacity 4 --session-checkpoint-mib 1024
```

The manager refuses an occupied GPU, freezes all worker sources and records the
changed command as an experimental session backend. Its reference recipe must
already have passed the existing base execution gate. The additional options do
not turn that base result into evidence for session correctness.

Send a stable `session_id` in the normal completion/chat body or use the native
session header supported by the pinned vLLM API. Cache identity is
`(cache_salt, session_id)` within this worker/model load. The serving gateway must
supply the intended user isolation scope; the plugin does not authenticate that
field. A session key selects a candidate checkpoint, while exact token-prefix
comparison decides whether any state can be reused.

The worker snapshots at the sealed boundary before the final prefill chunk. This
avoids assuming that rendered thinking or tool-call text equals raw generated
tokens. After the request finishes, capture is clipped to the native terminal
prefix, archive publication drains and the authority rewinds to its checkpoint.
The next request keeps fresh native sampling state and output limits. Full target
KV reconstruction plus GDN and draft context restoration happens after native page
zeroing and before attention metadata construction. If a prefix differs, the old
idle authority is evicted and the new prompt computes normally. This first version
keeps one prompt checkpoint per retained session, not arbitrary historical branches.

Entry count and retained checkpoint payload bytes are bounded. Admission failure
skips saving a checkpoint and preserves ordinary execution. Same-session requests
serialize while a checkpoint is active/retiring; other sessions remain eligible.
Idle entries are evicted under capacity pressure, with credits retained through
reader/cleanup fences. Native GPU pages are released between turns. Shared archive
capacity includes an additional `session_capacity` authority allowance. A live
request can protect both its checkpoint and current working frame, so repository
and GC reserve validation includes both. The checkpoint byte limit excludes Python
object overhead, authority indexes and archive/cache/transfer pools; it is not a
whole-process RAM quota. Timed expiry and cross-worker/restart recovery are absent.

Run the actual model gate only against the newly enabled, ready backend:

```sh
.venv/bin/python scripts/qualify_sessions.py \
  --engine-evidence artifacts/session_backend_02/server_state.json \
  --output artifacts/session_http_short_01
```

For the compressed-history gate, additionally pass `--input` pointing to a frozen
JSON file containing `input_ids`. The script preserves that input in full and
stores its SHA256. It compares cold and restored greedy token IDs, interposes an
unrelated request to recycle native pages, checks a changed-prefix miss, verifies
actual acknowledged reuse and waits for native page/retirement cleanup. Timing
observations in this functional gate are not concurrency or agent acceptance.

Telemetry includes acknowledged scheduler hits/misses, retained checkpoints and
payload bytes, byte-admission deferrals, restored token counts and terminal
checkpoint-rewind records. Idle authority slots are expected to remain allocated;
zero native request pages and no in-flight retirement are the between-turn criteria.
Eviction/shutdown must additionally drain the retained authorities.

The short gate (`session_http_short_01`, backend `ac566ea`) passes with 4096 prompt
tokens, a 3072-token imported checkpoint and 64 identical greedy output token IDs.
An unrelated request recycles native pages before import. A changed prefix correctly
evicts/recomputes and matches its separate cold control. Each request releases native
pages/reservations and drains retirement. One checkpoint retains 202761152 payload
bytes. Observed populate/reuse HTTP times are 5.622/2.321 s; this is a functional
observation, not repeated or concurrent performance acceptance. Thirteen GPU state
and assembly component tests also pass (`session_components_gpu_01.xml`).

The first enabled backend failed in native kernel warmup, which calls ordinary
`execute_model` without connector metadata. The adapter now starts session control
only after the same explicit post-warmup boundary used for authority capture;
formal requests still reject missing metadata. Five runner lifecycle tests pass.
The failed `session_backend_01` evidence is preserved.

The frozen Quill input `request_10.json` also passes in full (`session_http_long_01`):
134067 prompt tokens, SHA256
`8e5ea601d0bb63892e16938e061265ac650d36ef8a5fd20517e3c6b917057a6b`,
133120 imported tokens, 64 identical greedy outputs across cold control, checkpoint
population and restore. Restoring the already compressed frame and immediately
performing the next semantic query refresh succeeds. Changed-prefix recomputation
also matches its separate cold control without another session hit. Each call drains
native ownership to 0, with 1072/1073 physical blocks free (one null block).

Observed cold/populate/reuse HTTP times are 185.885/183.708/5.172 s. Native TTFT
falls from 182.686 s for population to 4.230 s on reuse. Native decode is 68.328
versus 73.852 tok/s over just 63 token intervals; both have 61.538% draft acceptance
and mean acceptance length 5.308. The main benefit is avoided repeated prefill,
not a 35-fold decode speedup. This functional gate is not an agent result or a
repeated performance comparison, and host development overlapped execution.

The separate concurrency/cancellation gate passes with the following command:

```sh
.venv/bin/python scripts/qualify_session_concurrency.py \
  --engine-evidence artifacts/session_backend_02/server_state.json \
  --output artifacts/session_http_c4_02 --concurrency 4
```

It populates independent checkpoints, requires real overlapping native request
ownership and exact import counts, and compares isolated continuations after the
concurrent restores against their original token streams. It also checks same-session
serialization and restored-decode cancellation/recovery. Concurrent token differences
are reported separately, since native batching can alter reduction order. Passing
this gate does not establish concurrent numerical quality or agent throughput.

The first 64-token fixture (`session_http_c4_01`) failed its overlap observation:
four native requests were outstanding, but periodic capture progress recorded at
most three active authorities. It remains a failed artifact. The 256-token fixture
in `_02` observes four active authorities and passes import counts, subsequent
isolated exact-token checks, same-session serialization (one active authority),
native cancellation/recovery, cache-salt isolation and idle LRU reclamation. The
cancelled restored request stops at accepted history 4096 with one sampled output
token, discards all eight in-flight verifier rows and retains its original 3072-token
checkpoint. Four checkpoints occupy 811044608 payload bytes, within the 1-GiB bound.
Explicit shorter-prefix requests then release all snapshots and archive slots;
native ownership/reservations also return to zero with 1072/1073 blocks free.

Concurrent streams differ from the serial controls after common prefixes of
86/90/253/67 tokens, with output lengths 256/256/254/94 versus serial 256 each.
The isolated continuations after this concurrency exactly reproduce all four
original 256-token streams. These results support checkpoint isolation and recovery;
they do not attribute the concurrent differences solely to rounding or establish
equal agent quality. Matched concurrent controls and executed task scoring remain
required.

The optional agent relay can send native `X-Session-ID` headers for an explicit
`--session-task` allowlist. IDs are deterministic within its unique run scope and
separate tasks and runs; request/response JSON bytes and task stopping/sampling
remain unchanged. This is benchmark-local namespace routing, not authentication.
`qualify_agent.py --session-evidence ...` requires passed long restore plus concurrent
cancellation gates from the same frozen backend before executing session-enabled
agent tasks. Actual session-enabled C2 tasks have now started in
`agent_sessions_c2_initial_01` (termenv and tomlkit) on frozen backend `ac566ea`,
after fresh tool and disconnect preflights. No score is available yet. The previous
non-session pair timed out with reward 0 for both tasks; it is retained in full.
Relay measurements are restricted to the new execution interval and task identities,
so preflight calls or earlier runs cannot enter the agent throughput statistics.
