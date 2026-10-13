# Multi-user serving

The `multi_user` profile runs authenticated users through one native vLLM engine.
Each user has a separate API key. The server derives session/cache namespaces
from the authenticated user, so two users can safely choose the same session ID.
The Python API process owns bounded request admission for the full response stream.

This adds a serving boundary around the pinned Q8_0/FP8/DFlash2 integration. Model,
GPU, compression and context restrictions in [support.md](support.md) still apply.
High client concurrency includes queued requests; `max_inflight` is the maximum
number admitted to the engine, not a claim that every connected client decodes at
once. Request fairness is not a token, time or storage billing quota.

## Configure

Start with `examples/qwen-q8-fp8-dflash.json`, set the local model paths, and add:

```json
{
  "multi_user": {
    "users_file": "users.json",
    "max_inflight": 4,
    "max_inflight_per_user": 2,
    "max_queue": 64,
    "queue_timeout_seconds": 60,
    "body_timeout_seconds": 30,
    "max_body_bytes": 8388608
  }
}
```

`users_file` resolves relative to the profile. Its format is:

```json
{"users": [{"id": "alice", "api_key": "replace-with-a-random-secret"}]}
```

Generate a different random secret for each user; keep this file outside source
control and readable only by the serving account. Keys are read at startup and
never copied into the launch manifest. Restart the server to rotate keys. Keep
the user ID stable to retain its namespace across key changes.

`max_inflight` must not exceed `max_num_seqs`. The launcher fixes
`--api-server-count 1` because admission and fairness are process-local. Additional
API processes or independent replicas need a shared external admission policy.
The launcher removes an inherited `VLLM_API_KEY` in multi-user mode, since tenant
keys are validated by this middleware. Rust API frontends are rejected.

```sh
python -m vllm_kvmem serve --config local-profile.json --run-dir runs/multi-user-01 --dry-run
python -m vllm_kvmem serve --config local-profile.json --run-dir runs/multi-user-01
```

Use a new run directory on each start. Client requests send
`Authorization: Bearer <user-key>` and may send `X-Session-ID` for exact-prefix
reuse. Send the full conversation on every turn. Namespace isolation does not
change the requirement for exact token-prefix matching.

## Active context, history and session lifecycle

The local profile `artifacts/multi-user-20261013/multi-profile-tensorcore.json`
uses the limits below. These describe that profile, not a reservation of these
resources for every user.

| Owner or resource | Isolation and limit in this profile |
| --- | --- |
| User | The API key resolves to a configured user ID; the server hashes that ID into `cache_salt`. A user can have at most 2 admitted requests, with 8 admitted globally and 64 additional queued requests. |
| Session | The lookup key is `(server-derived cache_salt, session_id)`. Equal session IDs from different users are separate. Turns within one session serialize, including requests that cannot retain a checkpoint. |
| Running request | Each HTTP request receives a fresh server-generated request ID. Its accepted-token authority, target window, recurrent state and draft context have independent ownership; model weights, native GPU pages and the archive arena are shared. |
| Active target window | Per request, selection budget `K=131072` and growth budget `G=65536` bound the dense target-attention window to at most 196608 token positions. Authority records contain 128 tokens each. |
| Session checkpoint pool | At most 4 checkpoint-capable session entries across the whole service, counting leased and retiring entries as well as idle entries; 1 GiB of accounted checkpoint payload across the worker's retained snapshots. Neither limit is per user. |
| CPU archive read cache | 16 GiB shared cache, with an additional per-authority cap of one K+G archive window. Eviction removes a cached copy; the authoritative archive record remains available. |

**Active context and history have different lifetimes.** During inference, the
GPU holds the selected target window plus the request's native recurrent/draft
state. Selection uses only that request's historical records, including its
protected sink/recent blocks. Older blocks can leave the GPU and later be
selected again. The authority retains accepted history as pre-RoPE BF16 K and
FP8 V: sealed blocks are written asynchronously to the shared NVMe archive;
incomplete tails and bounded staging remain on the host. Archive access requires
the authority namespace and a live slot-generation lease. Restoring a session
transfers its existing authority to the new request; it does not expose another
session's records or concatenate users' histories.

The scheduler reserves each admitted native request's possible future page
growth from its prompt length, output limit and speculative headroom, bounded
by K+G for target attention and including recurrent/draft page needs. When that
reservation cannot fit, the request waits for capacity. Thus **8 admitted
requests does not guarantee 8 simultaneous full 196608-token windows**. Short
requests may coexist in greater numbers than full-window requests. The native
logical context limit remains 262144 tokens; KVMem does not remove it. Global
and per-user request counts also do not guarantee equal GPU time or I/O bandwidth.
The native waiting queue can exhibit head-of-line blocking: a long request that
cannot reserve its full page budget may delay shorter requests behind it. The
120-second middleware queue timeout ends at admission; it is not a deadline on
native scheduling or the complete response. Clients should set their own overall
request deadlines. Mixed long/short latency remains a separate qualification item.

For a checkpoint-capable session, the worker captures an aligned, non-final
prompt boundary before the last prefill chunk. After completion or cancellation,
it drains outstanding work and rewinds the authority to that saved prefix before
advertising the checkpoint as idle. Native GPU pages are released after the
retirement acknowledgment; idle checkpoints retain host model state and their
archive authority, not the old native GPU pages. A subsequent turn must provide
the **full conversation** and have exactly the same token IDs through the saved
boundary. The worker rebuilds target KV on fresh pages and restores GDN/draft
state, then computes the remaining suffix. This is a saved prompt prefix, not a
server-maintained chat transcript or a snapshot of the latest generated reply.

The 1 GiB checkpoint budget accounts for token identity bytes, serialized window
geometry, and CPU GDN/draft tensor payload. It excludes the retained history
archive, GPU allocations, read cache, staging, and Python/native object overhead.
A snapshot can therefore be refused even with a free session entry. If all four
entries are busy, independent sessions can run cold without creating checkpoints;
if an idle entry exists, new sessions can evict the least recently retired idle
entry. A changed token prefix also invalidates its old checkpoint. Cold misses
recompute from the supplied conversation. Cache entries are shared across users
without per-user checkpoint reservations or a TTL, and are not restored across
worker/server restarts.

Storage limits are separate from those checkpoint limits. This profile leaves
`repository_blocks` at 4096 per authority. The shared arena has
`4096 * (8 active-authority slots + 4 session slots)` records; the observed model
uses 6 MiB per record, and its `archive.json` reports a **288 GiB logical arena**.
`repository_gc` is false, so active history is not aged out by background policy;
ordinary retirement, checkpoint rewind and eviction reclaim the appropriate
records after readers finish. The profile also leaves `nvme_preallocate=false`,
`nvme_max_bytes=0`, and `archive_io_max_bytes=0`: the arena has a finite logical
capacity, but startup does not reserve physical disk space, no additional byte
quota is configured, and the I/O queue uses its operation-count bound without a
configured byte bound. Logs and other run files are outside the arena limit.
These component bounds do not establish a global host-RAM limit or per-user
history/storage quotas; deployment still needs sufficient RAM and free disk.

The ownership and accounting above are implemented in
[session_control.py](../src/vllm_kvmem/session_control.py),
[capacity.py](../src/vllm_kvmem/capacity.py),
[prompt_checkpoints.py](../src/vllm_kvmem/prompt_checkpoints.py),
[session_worker.py](../src/vllm_kvmem/session_worker.py), and
[storage.py](../src/vllm_kvmem/storage.py).

## API and overload behavior

The multi-user entry allows authenticated `GET /v1/models`,
`POST /v1/chat/completions`, and `POST /v1/completions`. `GET /health` remains public.
Other endpoints, including administrative and stateful Responses endpoints, are
unavailable through this entry. Native single-user profiles retain their original
API. Remote deployments should terminate HTTPS at their existing reverse proxy;
the default listener remains localhost. This endpoint boundary follows the
[vLLM authentication limitations](https://docs.vllm.ai/en/stable/usage/security/#api-key-authentication-limitations).

Each admitted request produces one output sequence. Batched prompts and multiple
choices must be submitted as separate requests so admission cannot be bypassed.
One flat token-ID prompt remains supported. Server-generated request IDs prevent
clients from colliding with another user's native request.

An eligible user can run while another user is at its per-user limit. A full
queue or an expired wait returns HTTP 429 with `Retry-After`; clients should retry
with bounded exponential backoff. Disconnecting a queued request removes it;
disconnecting an active stream cancels its native request and releases admission
after cleanup. Invalid API keys, invalid inputs, oversized bodies, and overload
are reported separately from engine failures.
Incomplete request uploads time out with HTTP 408, including requests already
admitted, so a stalled client cannot hold its generation slots indefinitely.

## Validation and operating limits

Use [multi-user-testing.md](multi-user-testing.md) for concurrent streaming tests.
Run long-context compression/session checks separately: a short-request load test
does not prove long-context quality, decode-time compression, or unlimited history.
Keep the experimental Q8 Tensor Core option disabled unless your task-quality
and throughput comparison qualifies it.

The worker retains a bounded recent diagnostic history in serving mode. Native
trace files and model archives remain private operational data. Plan disk capacity
and log retention for long-running deployments; the per-component cache bounds do
not yet establish a global process RAM quota. The integration remains single GPU,
eager, synchronously scheduled, and version-pinned.
