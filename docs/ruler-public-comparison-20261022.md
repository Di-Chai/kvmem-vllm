# RULER public task-level comparison, 2026-10-22

Completion update: the active campaign subsequently finished all **260 cases**
at 14:02:45 Asia/Shanghai with an official task-macro score of **85.75/100**.
The fixed 206-case comparison below is retained as the requested historical
snapshot. Final task scores are available in the
[complete report](../artifacts/ruler-v1-256k-query-replay-20261022-r1/run/answers/report.md).
In the final set, multikey_2 scored 55, UUID multikey_3 scored 30 and QA2 scored
55. The cross-model/length limitations below still apply.

Fixed snapshot: **206 completed cases at 2026-10-22T13:25:48.315599+08:00** from `artifacts/ruler-v1-256k-query-replay-20261022-r1/`. The active run is untouched. The accompanying JSON records case IDs, score and source-result hashes.

Our run: Qwen3.8-27B, NVFP4 weights, FP8 KV, target 256K (observed 247,313–262,130 input tokens), selected budget 36,864 and generation reserve 16,384. Sampling temperature 1, output limit 8,192 and thinking budget 4,096. Each completed task currently has 15–17 cases.

Public columns below are transcribed from the [STRING authors’ 128K per-task table](https://github.com/HKUNLP/STRING#experiments). STRING modifies positional handling; these columns are not KV-compressed baselines. They are descriptive references with different models and inputs, not paired measurements.

| Task | Completed n | Ours, 256K | Qwen2-72B + STRING, 128K | Llama3.1-70B + STRING, 128K |
| --- | ---: | ---: | ---: | ---: |
| niah_single_1 | 17 | 100.00 | 100.00 | 100.00 |
| niah_single_2 | 16 | 100.00 | 100.00 | 100.00 |
| niah_single_3 | 16 | 100.00 | 100.00 | 100.00 |
| niah_multikey_1 | 16 | 100.00 | 95.20 | 97.20 |
| niah_multikey_2 | 15 | 60.00 | 86.40 | 71.40 |
| niah_multikey_3 | 16 | 18.75 | 56.20 | 81.20 |
| niah_multivalue | 16 | 96.88 | 94.50 | 96.60 |
| niah_multiquery | 16 | 100.00 | 97.15 | 99.60 |
| vt | 16 | 100.00 | 98.40 | 93.20 |
| cwe | 15 | 93.33 | 72.00 | 10.58 |
| fwe | 16 | 100.00 | 95.50 | 89.80 |
| qa_1 | 16 | 87.50 | 54.50 | 79.60 |
| qa_2 | 15 | 60.00 | 50.00 | 45.40 |

All values are on the 0–100 score scale. The 206-case weighted mean is 86.17; the provisional equal-task mean is 85.88. Neither is the final 260-case result. The public reported full-task means must not be compared with a different incomplete set as a paired delta.

The largest negative reference gaps occur in multikey_2 and multikey_3. Single needles, multiquery, VT and FWE are strong in this completed subset. CWE and QA1 have higher raw scores than these reference columns, without establishing cross-model superiority.

For the same model family, a [community Qwen3.8-27B FP8 report](https://huggingface.co/stoyani/Qwen3.8-27B-FP8-KVcal#ruler-long-context-128k) tests only single_1 and VT at 128K, 50 cases each, greedy and an output cap of 128. It reports single_1 = 100 across KV settings; VT = 61.2 (BF16 KV), 62.0 (uncalibrated FP8 KV), 66.0 (calibrated FP8 KV). Our corresponding completed scores are 100 (17 cases) and 100 (16 cases). Different prompts, limits, precision and inputs prevent attributing that difference to KVMem. These two-task results are not a complete RULER score.

This comparison cannot separate backbone, quantization, prompt and rolling-eviction effects. A same-model full-KV 256K control on the same frozen input IDs is needed to measure capability retained by KVMem.

Excluded from long-context comparisons: [CacheNotes](https://aclanthology.org/2026.eacl-long.309.pdf), section 5.1, and [GateSight](https://arxiv.org/html/2610.14754v1#S5.SS1), section 5.1, evaluate their reported RULER task suites at 4,096 tokens. Their separate long-context tests do not make those RULER scores 256K results.
