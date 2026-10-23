# Four original Key representatives per KV block

This opt-in experiment uses `query_score_mode=key_fps_4`. The production default
remains `mean_k`. It tests the same fixed twelve failure/control cases as the
previous MeanK, token-sum and 16/8 overlap answer experiments; this cohort is not
an estimate of complete RULER accuracy.

## Definition

For each original 32-token block, scoring layer and KV head, choose up to four
distinct accepted token positions. The first has maximum squared Key norm.
Each subsequent choice maximizes its minimum squared Euclidean distance to an
already selected Key. Distances use FP32 original content-frame Keys without
unit normalization. Ties choose the earliest original position. The cache holds
the selected original Keys, not cluster means. No Query, future text, answer or
target offset participates in representative selection.

For each Query row and Query head, use the matching GQA KV head and compute the
maximum of the representative dot products, divided by sqrt(head dimension).
Only then apply the existing native block `score_mask` and block softmax, average
layers/heads and sum Query rows. Invalid slots and maxima use negative infinity.
Partial blocks use all available tokens up to four distinct representatives.
The scoring domain and picker retain their original meanings; `transport_mask`
does not narrow the softmax domain. There is no neighbor expansion after picking.

Raw Key Euclidean coverage bounds maximum-dot-product error by Query norm times
the coverage radius (divided by sqrt(head dimension)); the bound may be loose
and does not guarantee preserved block ranking or answer accuracy. FPS max logits
are also a different scoring definition from global token-softmax/page-sum.

## Runtime and experimental controls

The initial qualification accepts only V2 multi with `graph_aggregation=sum`,
`graph_alpha=0`, no guided generation and no Query-history accumulation. Pressure
reselection continues to use cumulative attention. KV storage/loading/budget
remain in original 32-token blocks.

The owner-local derived index uses the existing shared bounded CPU Query-cache
reservation, including representative-position metadata. Record identity,
generation and accepted token count qualify reuse; rewind/close invalidate it.
The first Query builds a cold index from accepted archived Keys, so its archive
reads, selection and copying must be included in timing. This first version does
not claim asynchronous prefill-time index construction or a serving speedup.

The live prototype uses the general archive reader, which reads/clones K and V
for all 17 archived attention/MTP layers before FPS takes K from 16 target
layers. Near 256K history this is about 8.49 GiB of raw payload versus 4.00 GiB
of necessary K, before additional packing and FP32 FPS scans. The reader has
parallel gather workers, but these do not parallelize FPS automatically.
Independent offline construction takes 52--55 seconds, including 34--38 seconds
in FPS itself; its K-only input and smaller batch working set differ from the
live path, so these numbers do not isolate the archive reader's contribution.
No live substage profile or warm-owner latency measurement has been completed.

The answer experiment uses the same complete 256K fixtures, per-case seeds and
sampling as previous controls: K=36,864, G=16,384, sink=1,312, recent=0,
NVFP4 weights, FP8 KV, MTP(k=4), chunk=2,048, batch limit=16,424 and serial
requests. It preserves the 8,192-token answer cap and 4,096 thinking budget.
Graph propagation, guided Query and physical prefill coalescing stay off.

Each run freezes its source and profile, checks original input/reference hashes,
records actual generated answers and accepted-history evidence, independently
replays the native picker, and audits GPU cleanup. Existing answers are never
overwritten or regenerated for a lower score. Fixed-Q/K offline recall analysis
is reported separately from actual model-generated answers.

## Results

Qualification and reporting checks passed 165 CPU tests and two CUDA tests, including the full
8,192-block scoring geometry's <=512 MiB temporary GPU bound. Three additional
FP8/GQA CPU/GPU fixtures selected identical token offsets and Keys; their largest
page-score difference was 1.2e-7.

Fixed-Q/K CPU replay of three previously failed cases recovered all target
values. These are original archived operands and native picker results, not
newly generated answers:

| Case | MeanK target rank | Token softmax/page sum | All-token block max | FPS-4 |
|---|---:|---:|---:|---:|
| 0011 | 6,725 | 18 | 15 | 6 |
| 0012 | 4,185 | 6 | 6 | 5 |
| 0015 | 7,839 | 11 | 11 | 13 |

MeanK coverage is 0/3 and FPS coverage is 3/3 on these fixed operands. The
diagnostic reads mmap captures on CPU; its timing is separate from the live
archive/cache path and overlapped model startup. Evidence:
`artifacts/query-key-fps-fixed-operands-20261023-r1/summary.json`.

The twelve actual-answer cases completed with frozen-source, original-input,
seed/sampling, native-picker and reference-answer hash audits passing. Owned
model processes and GPU allocations were released. Evidence and the independently
recomputed final analysis are in `artifacts/ruler-key-fps-4-20261023-r1/report.md`
and `analysis.json`.

| Scorer (graph off) | Correct answers | Full target-key recall | Full target-value recall |
|---|---:|---:|---:|
| MeanK | 3/12 | 4/12 | 3/12 |
| 16/8 overlapping MeanK | 7/12 | 8/12 | 7/12 |
| All-token softmax/page sum | 9/12 | 12/12 | 9/12 |
| Four original Keys / block max | **11/12** | **12/12** | **10/12** |

Historical full-context native vLLM answered these same twelve cases correctly;
its concurrency differed, so it is only an answer reference. The cohort contains
nine prior MeanK failures and three successful controls, not a representative
complete-RULER sample. FPS improves eight answers relative to MeanK, two relative
to token-sum, and five relative to overlap, but regresses overlap case 0009.

Both incomplete values cross a 32-token boundary. Case 0009 selects the first
3/7 value tokens but omits the remaining four and answers incorrectly. Case 0003
selects 6/7 and nevertheless answers correctly; it counts toward answer accuracy
but not full value recall. The final answer alone cannot establish why it
recovered the missing digit. All ten fully recalled values were answered correctly.

This prototype is not a latency optimization. Cold index construction has a
118.03-second median, and representative transfer/scoring/synchronization a
6.91-second median. The whole request median is 174.68 seconds, compared with
83.93 seconds for the historical token-sum experiment (output lengths differ).
The bounded representative cache occupies 1.018 GiB per owner; all twelve
requests used distinct owners and had no cache hits. Warm reuse is unmeasured.
The first case overlapped the offline CPU diagnostic, so timings also have a
recorded CPU-load difference. The mode remains opt-in.

## Cross-block completeness: proposed follow-up, not implemented

The two remaining incomplete values are present in accepted archived history
and allowed by both recorded masks. Their anchor blocks rank 3 (0009) and 6
(0003), while the next blocks rank 5,220 and 3,233; the actual optional cutoff
is rank 1,111. This is a low-scoring continuation problem, not missing archive
data. Raising adjacent scores alone cannot ensure both blocks survive the picker.

Keep four-Key scoring as the locator, and evaluate atomic groups inside the
existing KV budget. A simple first mechanism test would give every anchor the
same bounded neighborhood (one preceding and one following logical block),
rank the group by its anchor score, and admit the complete group only if its
new, deduplicated blocks fit. No group would be chosen from answer labels.
This deliberately changes the retrieval unit, not KV storage or Q/K scoring.
It would be a separate switch/experiment; this completed run has no expansion.

The repository already contains `KvMemStore::pick_repository_semantic_groups`
in `third_party/qw3/src/kvmem_store.cpp`: it preserves mandatory blocks,
charges overlapping groups by newly selected blocks, and skips groups that
cross missing history or do not fit. It is not yet exposed through the current
Python preview path, and admissibility/transaction integration needs tests.
Scoring masks, transport masks and read permissions must retain separate roles;
in particular transport_mask must not become a neighbor-loading mask.

A three-block neighborhood can reduce 1,111 available optional blocks to about
370 isolated anchors in the worst case. Completeness therefore competes with
breadth, even without extra QK products. After that mechanism check, use winning
representative offsets and accepted-text sentence/record boundaries to form
smaller bounded groups. Do not recursively expand newly added neighbors, cross
owner/message boundaries or historical holes, or consume unaccepted future text.
Grouping must happen before admission; adding neighbors after a full-budget
selection would violate the comparison.

This separates precise retrieval from the surrounding content supplied to the
model, analogous to the sentence-window approach documented by
[LlamaIndex](https://developers.llamaindex.ai/python/examples/node_postprocessor/metadatareplacementdemo/).
That text-retrieval example does not establish an accuracy gain for cached KV.
The next evaluation should report both complete-span recall and actual answers,
how many other anchor blocks were displaced, and selection/load latency, all at
the same K budget. No grouped-recall answers have been generated in this run.
