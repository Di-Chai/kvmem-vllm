# Qwen3.8-27B 256K evaluation on the cloud RTX 4090

Evaluation started on 2026-10-23 and finished on 2026-10-24.

The corrected C1 profile passed an actual 262144-token logical history test.
CPU history was extended through 65568, 245792 and 262144 accepted tokens,
with two real restores of the previous exact prefix. The sampled GPU peak
was 22503 MiB. This establishes synthetic storage/admission/restore capacity;
task completion and GC throughput are evaluated separately below.

## Serving profile

| Setting | Resolved value |
|---|---|
| Model | sokada4/Qwen3.8-27B-GPTQ-Int4, revision 2232b67a2ef5480e97c4450838a712de1346eccc |
| Download source | hf-mirror; all checkpoint files previously verified |
| GPU and concurrency | Cloud RTX 4090 24 GiB only; one sequence/request at a time |
| Logical model context | 262144 |
| KVMem selection budget K | 36864 |
| Generation budget G | 16384 |
| CPU history authority | 12 GiB; 8196 repository records, block size 32 |
| CPU recurrent/checkpoint pool | 2 GiB, P/M terminal capture |
| Model compute / KV | BF16 / FP8 E4M3 |
| Speculation | Native MTP, one draft token; shared embedding/head loading |
| Execution | Eager; async scheduler and native prefix caching disabled |
| GPU memory utilization | 0.86 |
| Native pool | 45 pages; minimum bounded C1 working set requires 41 |
| Prefill / replay | 128 / 128 tokens; resolved scheduled batch 130 |
| MeanK | torch_reference with one-layer index transfer and bounded workspace |
| Historical GC | statistical/delete, poll 10 ms, scan128/delete8 |
| Block graph / native attention producer | Disabled in this evaluation |

The GPU budget is K+G=53248 physical attention tokens plus the existing
recurrent/null-page requirements. Logical positions and block-table reach remain
262144. The opt-in bounded capacity adapter retains the native worker byte
accounting, allocation and capacity checks; it changes the demand attributed to
bounded attention storage. See [the adapter contract](bounded-physical-kv-capacity.md).

The full CPU history does not imply all history is simultaneously attended to
on the GPU. Full-attention KV is archived, while the selected window is assembled
and the hybrid model's recurrent state is preserved by the checkpoint protocol.
No model-weight CPU offload was used.

## Capacity test and the discovered failure

| Prompt | Output | Accepted logical history | Durable terminal KV | GPU window | Time |
|---:|---:|---:|---:|---:|---:|
| 65536 | 32 | 65568 | 65567 | 36927 | 78.04 s |
| 245760 | 32 | 245792 | 245791 | 36927 | 229.88 s |
| 262112 | 32 | 262144 | 262143 | 36927 | 25.99 s |

Every extension retained all previously accepted IDs. The checks require the
exact uint32 prefix hash, complete history records and CPU bytes, current terminal
M checkpoint, successful matching restore event and session-hit count. After
the final stage, the owner, reservations, native pages and CPU admission credits
were released. GC retired zero blocks, as required for this complete-history
capacity fixture.

The last accepted output token normally has not undergone another forward,
so this length-capped run has 262143 durable KV tokens for 262144 accepted
logical tokens. This distinction is checked against the actual terminal record,
not assumed for cancellations or arbitrary failures.

The final CPU authority held 9125691392 bytes of sealed KV; the remaining partial
tail belongs to the terminal checkpoint. Engine RSS peaked at 13632.80 MiB,
including archive/index/checkpoints and runtime. The cgroup also includes shared
model file cache, so its usage is not equivalent to live archive bytes.

The first attempt passed 240K but failed near the final 256K boundary: MeanK's
all-layer GPU einsum attempted another 512 MiB allocation with 510.69 MiB free.
The sampled GPU peak before that failure was 23995 MiB. CPU storage and native
page admission were still valid.

The reference scorer now packs/transfers one index layer at a time and uses
contiguous grouped-query matrix multiplication. Each softmax still covers all
eligible history blocks. For this model, the main explicit scoring arrays total
about 56 MiB; library workspaces and other owners are additional. FP32 reduction
order changes, with old-formula tests at rtol2e-6/atol1e-7 and clear-margin ranking
checks; arbitrary near-tie ordering is not guaranteed. Seventy-five CPU tests and
independent review passed before the identical GPU capacity fixture was rerun.
See [the workspace contract](mean-k-reference-workspace.md) and
[the complete capacity report](benchmarks/qwen38-4090-256k-capacity-20261023.json).

## Paired GC throughput

The predeclared evaluation uses five alternating OFF/DELETE pairs, a frozen
36864-token prompt and 512 output tokens after a matching warmup on each fresh
server. K/G/logical/CPU budgets are the values above. The 1% throughput-loss
criterion is evaluated only after all five pairs.

The five-pair result is **inconclusive** against the predeclared 1% margin.
Median paired throughput loss was -1.3633% (negative means faster with GC);
the descriptive 95% bootstrap interval was[-2.4302%, +2.7937%].
All ten measured outputs were identical, with three rejected MTP draft tokens
per request. Every DELETE sample witnessed four candidate CPU frees during
decode with subsequent accepted tokens; OFF retired none.

| Pair | OFF token/s | DELETE token/s | Loss |
|---:|---:|---:|---:|
|1|27.7738|26.9979|+2.794%|
|2|27.2400|27.7043|-1.704%|
|3|27.4033|27.7769|-1.363%|
|4|27.4511|28.1182|-2.430%|
|5|27.5162|27.7795|-0.957%|

A post-hoc diagnostic limited to accepted tokens32..480 also covered each
candidate release, with median loss-1.4254% and range[-2.2045%,+2.9198%].
It is diagnostic, not a replacement noninferiority claim. OFF saves terminal
M37375; DELETE defers M and retains P36832, so subsequent replay is excluded.
The DELETE aggregate CPU release delta is16, of which exactly4 candidate frees
are witnessed inside decode. See [the complete report](benchmarks/qwen38-4090-36k-gc-throughput-20261024.json).

 The fixture publishes synthetic eligibility for
four generated blocks, with one retained result block. It must demonstrate
retirement and CPU release inside decode, followed by further accepted tokens;
shutdown cleanup cannot satisfy that condition. This measures the policy and
reclamation overhead under a 36K active window, without a semantic-estimator or
256K-history retrieval-throughput claim.

## DSH task protocol

The original expense-tracker and CSGO prompts are preserved byte-for-byte in
all three task attempts. Each uses a fresh owner, logical262144/K36864/G16384,
thinking disabled and the complete system/tool frame pinned. Sampling remains
temperature0.6/top_p0.95/top_k20/min_p0/presence0/repetition1/seed1000.

The first expense and CSGO attempts used logical output cap16384, one native
segment, idle900000 ms and turn budget1800000 ms. The subsequent CSGO retry
used cap32768, up to two serial16384-token native segments, idle1800000 ms and
turn budget2700000 ms. The native G budget stays16384 in both configurations.
A turn deadline triggers public cooperative cancellation and flush; natural
completion and budget termination are distinguished. Compaction is disabled.

The bridge now caps output to the actual remaining logical room instead of
rejecting every request with less than a full G available. The earlier32K
expense/CSGO failures exceeded that old preflight by111/59 tokens respectively.

The expense task naturally completed in 373.865 seconds with 12 committed
model replies,11 tool calls,8705 accepted output tokens and11 exact-prefix
reuses. Its largest prompt was13474 tokens. The original index.html/app.js
passed all 33 browser state checks, with no page/console/request errors.
[The unmodified artifact](benchmarks/samples/expense-tracker-256k-20261024/index.html)
and its browser plan/results are preserved.

The initial CSGO run created index.html and styles.css, then reached16384
output tokens inside its game.js tool arguments. Native finish_reason=length
and the incomplete outer call caused strict parsing to reject execution.
The browser confirmed game.js404. This was the DSH logical-response cap,
set equal to G by the evaluation setup; KVMem itself could accept a longer
logical response across multiple G-sized native requests.

A fresh rerun retained the original prompt and all serving budgets, with
DSH cap32768, at most two serial16384-token segments, idle1800000 ms and
turn budget2700000 ms. Its first147-token response contained malformed JSON
in todo_write, so schema validation stopped the trajectory before any tool
or segment continuation. This configuration has not yet demonstrated a
successful long tool completion. The [incomplete first artifact](benchmarks/samples/csgo-incomplete-256k-20261024/index.html)
is labelled as such and is not a playable game.

[The full DSH report](benchmarks/qwen38-4090-256k-dsh-20261024.json)
preserves both failures and the success, the distinct logical output caps,
tool/restore counts and release evidence. All owners were released, CPU
admission reservations and native working pages cleared, services stopped,
and the cloud RTX 4090 settled at 0 MiB. Native per-task GC retirements remained0.

 No genuine DSH semantic eligibility producer is
installed, so mode=delete alone does not establish semantic GC effectiveness.
