# Graph-guided cleanup feasibility pilot

This directory implements the offline preparation, source audit, RTX 4090
shadow KV intervention, native System One judgment, serving, and recovery pilot.
It never executes commands from source traces or changes online KV policy.
Human semantic labels are deliberately left unfilled.

```sh
python3 -m unittest discover -s experiments/graph_cleanup -p 'test_*.py' -v
python3 experiments/graph_cleanup/prepare.py \
  --bundle artifacts/agent-traces-handoff-20261102-r2 \
  --output artifacts/graph-cleanup-NEW
python3 experiments/graph_cleanup/audit.py \
  --bundle artifacts/agent-traces-handoff-20261102-r2 \
  --dataset artifacts/graph-cleanup-NEW \
  --output artifacts/graph-cleanup-NEW/audit
```

Each output directory must be new. The original sources are checked by SHA-256
and exact JSON field/character coordinates. Source eligibility follows the
existing structural normalizer, excluding framework reminders embedded in tool
results. Error controls require observed diagnostic text, not an `Error` symbol
in source code or an arbitrary fragment inside a failed tool result.

The pilot selects six groups of four candidates. Early snapshots include
complete eligible Steps within 60,000 source characters. Actual failure controls
may need later, explicitly left-censored windows plus task constraints. These
re-rendered research histories exclude framework/tool schemas and do not claim
to reproduce original agent prompt tokens. All candidate evidence is available
at the frozen checkpoint. Exact-duplicate samples can share a text cluster;
their positions differ but they are not independent task samples.

Graphs use observed tool IDs and computed lexical/entity associations. No
attention, semantic dependency, or supersession ground truth is fabricated.
Rankings and construction cost can be inspected before independent quality
labels exist, but accuracy, recall, false-safe rate, and safe-to-free bytes
remain unmeasured.

`token_mapping` supports tokenizer offsets when an actual tokenizer is supplied.
It distinguishes logical blocks and fully contained blocks and reports other
information units touched by outward rounding. The GPU run tokenizes the replay with the actual model tokenizer and validates
logical-to-physical mapping against native slot mappings at every attention layer.

The audit exports Full/Reduced **text-ablation** inputs and fixed six-question
probes. Re-prefilling reduced text is a separate intervention from removing
already-computed KV. Actual KV impact requires matching original Query positions,
frozen historical KV/recurrent state, separate Attention/Layer outputs, and
zero-deletion/stability controls. The qualified model's recurrent states can
retain historical information after attention KV deletion.

The native `/v1/systemone` endpoint returns categorical decisions, not arbitrary
256–1024-token answers. A generative six-probe observer and a native consistency
judge must be named and measured separately. The long-output serving matrix and the native categorical serving matrix are
measured separately. Recovery files contain native attention K/V only; they do
not contain recurrent states or measure live physical-page installation.


## GPU workflow

Use a new output directory for every run. The scripts depend on the qualified
remote Qwen3.5-family 64-layer hybrid model and pinned vLLM installation; the
geometry checks intentionally fail for incompatible models.

```sh
python experiments/graph_cleanup/gpu_study.py \
  --dataset artifacts/graph-cleanup-4090-20261105-r4 \
  --output paired-NEW --limit 24 --max-tokens 768
python experiments/graph_cleanup/recovery_study.py \
  --archive paired-NEW/recovery/attention-kv.pt --output recovery-NEW
python experiments/graph_cleanup/service_study.py judge \
  --inference paired-NEW --output judge-NEW
python experiments/graph_cleanup/service_study.py matrix \
  --inference paired-NEW --output serving-NEW
```

`gpu_worker.py` excludes selected historical physical page IDs from the
FlashInfer attention read. It preserves the original absolute positions,
precomputed historical K/V and recurrent state. Full and Reduced use the same
backend and full query batch shape; zero-deletion/native and repeated-run
controls precede all measurements. Only the fixed 148 query tokens contribute
to the paired output metrics. Generated answers are allowed to diverge.

Attention output is measured before gate/output projection, with all 24 heads
concatenated for each token's relative L2. Transformer output is the float32
sum of the delayed residual and hidden output at each decoder layer. Statistics
are across tokens, then explicitly across layers. These are separate metrics.
The first three recurrent layers precede the first attention intervention and
provide a structural zero-change check.

The shadow experiment retains the actual storage for audit and reports zero
physically freed bytes. Excluded attention bytes are an intervention size,
not a measured memory saving. FP8 attention K/V archives exclude recurrent state.
Re-prefill checks reconstruct the complete original prefix from position zero;
this does not establish reconstruction from isolated candidate text.

`continue_gpu.py` is a site-specific coordinator for the 2026-11-05 run. It waits
for the owned paired process, runs copy/file recovery, starts an isolated native
loopback API, performs judgments and both serving matrices, then stops only
that API's process group. Source commands are never executed. The previous
service is restored separately after checking final results.

`analyze_gpu.py` audits token identities, historical hashes, layer/query coverage,
and repeated measurements, then exports candidate/layer/ranking/serving CSVs.
`plot_gpu.py` emits PNG and PDF figures using matplotlib. Native judge scores
are model-assisted estimates, not independent labels; Precision/Recall and
false-safe remain null until a separately reviewed dataset is available.


## Current experiment findings and review

The final report is under `artifacts/graph-cleanup-4090-20261105-gpu-r1/REPORT.zh-CN.md`.
While the serving matrix is running it explicitly identifies itself as a snapshot.
The GPU pairs, page-count sweep and recovery components completed. The initial
native semantic judge confused some citation changes with fact omissions. The
citation-normalized revision still failed a cross-field retention control and
misclassified one literally identical progress field. Its scores are retained
as **unvalidated diagnostics**, not safety labels. Both revisions and their
controls remain available.

`judge_revision.py` normally stops at a failed control. The explicit
`--continue-recorded-controls` mode records diagnostic results without changing
the failed status. `service_study.py matrix --resume` preserves completed cells
and renames unfinished cell attempts before retrying. It never mixes two
serving loads in one measured cell. `progress.py` reads live counters only;
`snapshot_serving.py` streams a checkpoint of completed cells.

Native input calibration detects prefix-length cycles and tries recorded neutral
trailing whitespace to reach the exact target usage. Calibration calls are
outside measured cells. Resuming skips fully completed input-length groups and
preserves the prior calibration attempts.

`restore_previous.py` starts the saved prior service configuration after the
experiment API has stopped; it refuses to replace another listener. It is
site-specific and is only invoked after the measured jobs are complete.

`verify_artifacts.py --root RUN` checks all 120 generative configurations and
20 native categorical configurations against individual responses, token counts,
event times, aggregate calculations and recorded failures. Its `--allow-partial`
option audits completed checkpoints without claiming a complete matrix.

After restoration, `freeze_run.py` checks that the recorded inference sources
and weight inventory have not changed, hashes the model/tokenizer and records a
manifest of the finished remote run. Run `verify_download.py --root LOCAL_RUN`
before regenerating local analysis: it checks the downloaded files against that
manifest and lists native KV archives deliberately retained only on the server.
Locally regenerated reports, figures and review files have a separate manifest.
