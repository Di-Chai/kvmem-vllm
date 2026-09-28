# Experimental exact prompt sessions

This path has passed its short actual-model HTTP/GPU token-equality gate.
Compressed-state continuation, concurrent cancellation and agent performance are
still pending. It is disabled by default and is separate from native streaming-input
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

The separate concurrency/cancellation gate is ready for GPU execution:

```sh
.venv/bin/python scripts/qualify_session_concurrency.py \
  --engine-evidence artifacts/session_backend_02/server_state.json \
  --output artifacts/session_http_c4_01 --concurrency 4
```

It populates independent checkpoints, requires real overlapping native request
ownership and exact import counts, and compares isolated continuations after the
concurrent restores against their original token streams. It also checks same-session
serialization and restored-decode cancellation/recovery. Concurrent token differences
are reported separately, since native batching can alter reduction order. Passing
this gate would not establish concurrent numerical quality or agent throughput.
