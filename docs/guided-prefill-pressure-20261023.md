# Guided Query at prefill pressure boundaries

The requested experiment replaces cumulative-attention pressure scores with a
generated private Query. The final explicit Query still uses the original public
Query Q and V2 multi max propagation (alpha 0.2). Prefill pressure does not add
graph propagation. Enable this opt-in policy with `guided_reselect="prefill"` and
a positive `guided_query_tokens` cap. The qualified configuration uses cap 512,
asynchronous refresh, recent=0, and query-history accumulation disabled.

At an ordinary prefill pressure boundary, reserve private scratch inside K+G,
snapshot native recurrent/speculation state and the acceptance counter, and
generate against the current accepted active window. The upcoming public replay
tokens and the final benchmark question are not privately replayed or read.
Naturally terminated, non-structured Query Q scores the accepted historical
MeanK index using the existing native score domain; the original pressure picker
then keeps sink/mandatory evidence and selects up to K. The target recurrent
state is restored before selection/assembly. All private rows remain excluded
from public output, accepted attention, graph construction and history append.
An accepted-history/statistics witness is checked on each transaction. The
qualification additionally checks bitwise recurrent/counter restoration.

Capacity, empty-output, structured-output and unterminated-output failures are
explicit cumulative-attention fallbacks. Reports separate generated Query
attempts, consumed/published tokens, effective score source and fallback reasons.
The output cap alone is not a valid natural termination. There is no persistent
private-Query cache or dependence on future input. Final explicit Query replay
does not start private generation in this mode.

The original `ruler-pooling-queue-20261022-r1/max_guided` arm generated guidance
at the final Query boundary instead. It was stopped after 182 completed cases
when the user corrected the intended experiment. Its files and hashes are
preserved; those results cannot answer whether generated Query scores improve
prefill pressure. Both completed no-guidance control arms remain valid.

The corrected root is `artifacts/ruler-guided-pressure-20261023-r2`. It reuses
the completed max and Query-only controls, freezes a new runtime, and separately
qualifies the first eight original 256K cases before launching 260 fresh cases.
Inputs, per-case seeds, sampling, NVFP4/FP8 KV/MTP4, eight-owner admission,
K=36,864/G=16,384, batch 16,424 and ordinary 2K prefill are unchanged.
The final audit requires generated Query scores at pressure boundaries and
zero private generated tokens in the final original-Query selection.

Current CPU verification: 210 related tests passed, covering pressure headroom,
unread-input exclusion, preservation of raw final Query Q, native selection,
unchanged accepted attention, phase-specific protocol rejection, cancellation,
bounded resources and existing replay behavior. GPU qualification and complete
RULER results are recorded under the corrected root.

Eight-way GPU qualification passed: all eight complete 256K inputs retired and
released their owners. There were 112 prefill pressure attempts: 111 published
generated Query scores and one empty-output fallback. All 112 passed exact
native recurrent/speculation-row and acceptance-counter rollback checks. All
eight final Query selections retained the original public Query and native max
propagation, with zero generated private tokens. The full fresh 260-case arm
was then started; qualification scores are not a complete RULER result.

The first corrected qualification attempt (`r1`) is preserved separately. Its
inherited verification environment flag was removed by the serving launcher's
environment cleanup, so it did not qualify. Version `r2` uses the explicit
`serve --verify-guided-pressure` diagnostic argument, recorded in the launch
plan. The complete benchmark does not enable that extra diagnostic work.
CPU evidence and the qualification receipt are in
`artifacts/ruler-guided-pressure-monitor-20261023-r2/validation-tests.xml` and
`qualification-verified.json` in the same directory.
