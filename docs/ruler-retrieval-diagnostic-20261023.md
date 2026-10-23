# RULER Query retrieval diagnostics

This is a separate, instrumented experiment. It does not resume the interrupted
guided-pressure campaign or enable its stopped 20-minute monitor.

The frozen attempt is `artifacts/ruler-retrieval-diagnostic-20261023-r4`, backed
by the data disk. Earlier r1–r3 directories contain preparation-only validation
failures; no model was launched from those preparations.

## Frozen inputs and controls

The cohort contains all nine failed `niah_multikey_2` cases from the completed
Query-only/sum controls, plus successful cases 0000, 0013 and 0005. The original
256K token-ID files, their hashes, the original public Query spans and seeds are
reused. Gold answers and target positions are read only by post-hoc analysis;
the server receives neither diagnostic annotations nor future input.

Both arms retain NVFP4 weights, FP8 KV, compiled MTP4, 2K prefill chunks, G=16K,
sink=1312, recent=0, cumulative-attention pressure and Query alpha=0. Requests
are serial with `max_num_seqs=1`; ordinary physical prefill coalescing is off.

* Compressed: K=36,864, original repository capacity.
* Complete-prefix control: K=245,760/G=16,384, allowing a 262,144-token working
  frame. Repository capacity is increased to 32,768 records to accommodate
  checkpoint and live frames. The diagnostic is taken **before** final Query
  selection. The audit requires no prior pressure event, original dense Query
  positions and native length equal to the original Query end. The remaining
  seven chat-template suffix tokens have not yet been admitted at this observation;
  no future public input is read early.

Only one output token is requested: all observations precede answer generation.
G remains 16K. A fixture test verifies for all 20 original multikey-2 inputs that
this output limit preserves compressed-arm prefill pressure boundaries and that
the complete-prefix arm has no pressure boundary before Query. These outputs
are not scored, and the experiment makes no throughput claim.

The full arm is a control using the same KVMem instrumentation. It is not the
previous unmodified-vLLM benchmark.

## Three distinct observations

1. `native_attention.npy` reconstructs attention for the original Query rows
   over the actual native FP8 working cache, preserving its K scale, Q codec,
   dense RoPE and complete causal normalization. Heads are averaged; layers and
   original Query rows remain separately available. Missing historical pages
   in the compressed working frame have no native-attention observation.
2. `means.pt`, `queries.pt` and `raw_keys.npy` retain the original content-frame
   operands. `archive.json` records all accepted logical pages, score and
   transport masks, production MeanK scores and a fixed-K=36,864 picker preview.
3. Post-hoc replay uses the exact same Q/K and eligible history to compare
   `softmax(Q @ mean(K_page))` with global token softmax followed by **sum**
   within each page. This intentionally omits RoPE, matching the content frame
   of MeanK, to isolate averaging. The separate max-token-logit proxy is only a
   diagnostic; it is unrelated to max graph propagation. A fourth observation,
   `token_max`, takes the maximum token probability within each page **after
   the same global token softmax**. Unlike page sum, those page maxima do not
   sum to one; this distinguishes block max pooling from normalizing page maxima
   with a different denominator.

Production heads/layers are averaged and Query rows summed. No page receives an
independent softmax denominator. The native-attention domain includes fixed
pages and visible Query tokens, so its probabilities must not be described as
the same distribution as the content-frame retrieval probe.

Paired analysis crosses compressed/full Q with compressed/full K. It additionally
reports Query-key-token-only aggregation, target K byte equality/cosine and Q
cosine by layer. These are counterfactual score/coverage experiments, not new
model generations or corrected benchmark scores.

## Lifecycle and reproducibility

`--retrieval-diagnostics` is an explicit diagnostic launcher switch, disabled by
default and incompatible with guided generation. Capture is synchronous,
expensive and bounded by the frozen input; it must not be used for throughput
measurement. Original pages stay pinned during reads; the existing selector and
commit/rollback remain authoritative. Each owner is retired before the next
case, and each owned model process group is cleaned before the next arm.

Frozen source hashes are in `runtime-proof.json`; profiles, inputs, requests,
responses, per-owner captures and cleanup receipts are retained. No completed
historical answer is overwritten or regenerated. Plot dependencies are isolated
under `/tmp/kvmem-retrieval-plot-deps`, outside the model environment.

Scripts: `diagnose_ruler_retrieval.py` prepares/runs captures;
`analyze_ruler_retrieval_diagnostic.py` replays and audits paired operands.
The diagnostic math tests cover GQA, full-domain normalization, masked pages,
partial pages and a fixed example in which averaging reverses a strong token
match. Related private-Query, score-domain and async-refresh regressions pass.

The final paired report belongs in the attempt's `analysis/summary.json`.
Partial compressed-only reports are explicitly separate under
`analysis-partial/`; they do not establish the effect of changing prefill context.

## Answer validation

`scripts/test_ruler_retrieval_answers.py` follows the capture/analysis job with a
separate 12-case comparison. It preserves the original complete 256K inputs,
public Query spans, seeds and 8192-token output limit. Both arms use serial
requests with the same 36,864/16,384 budgets, original cumulative-attention
pressure and no Query graph propagation. Fresh MeanK answers are compared with
global content-token softmax followed by page sum. No neighbor expansion or
answer-dependent selection is used. This failure-enriched cohort cannot be
reported as overall RULER accuracy.

The token scorer reads accepted generation-qualified archived FP8 K in bounded
128-record groups. It retains the native score mask, mandatory sink, selector,
fenced assembly and graph-version validation through commit. Its explicit
`--experimental-query-token-scores sum` switch is disabled by default. It is a
slow diagnostic implementation, not an optimized production scorer.

Full original responses, final-answer-only official scores, accepted-token
accounting, actual committed-window coverage, hashes and owned-GPU cleanup are
saved separately. The answer controller requires completed and cleaned Q/K
capture, rather than waiting for unrelated post-hoc analysis. Remaining analysis
runs with one low-priority CPU thread and no CUDA context. The recovery control
source and original versions are preserved under `recovery-20261023-r2`; frozen
inference runtime, inputs and sampling remain unchanged. Completed analysis is
reused, incomplete postprocessing files are preserved before recomputation, and
the postprocessing device is declared per case. No recurring monitor or
automatic conversation wakeup is installed.

The completed answer attempt is
`artifacts/ruler-retrieval-answer-20261023-r1/report.md`: MeanK answered 3/12,
token page-sum answered 9/12. Six same-page failures were corrected; the three
remaining failures lost numerical suffixes on adjacent pages (1, 4 or 5 tokens).
All nine complete-value retrievals in the token arm produced correct answers.
Both arms and the paired analysis passed final audits and released their owned
GPU processes. The original inputs, complete responses and hashes are retained;
this selected failure cohort is not an overall RULER accuracy estimate.
