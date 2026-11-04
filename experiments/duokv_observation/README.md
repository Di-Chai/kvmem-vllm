# DuoKV offline observation study

This implementation analyzes the supplied `agent-traces-handoff-20261102-r2`
bundle without running recorded commands, model inference, or network requests.
It writes to a separate output directory and verifies the bundle before and
after analysis. Automatic relationships are evidence-backed candidates, not
human labels or causal deletion judgments.

```bash
python3 -m pip install -r experiments/duokv_observation/requirements.txt
python3 experiments/duokv_observation/run_study.py \
  --bundle artifacts/agent-traces-handoff-20261102-r2 \
  --output artifacts/duokv-observation-20261103
python3 -m unittest discover -s experiments/duokv_observation -p 'test_*.py' -v
python3 experiments/duokv_observation/validate_output.py \
  --output artifacts/duokv-observation-20261103 --write-audit
```

The full output README and seven reports contain metric definitions,
denominators, exclusions, examples, and limitations. The output includes a copy
of all code and exact plotting dependency versions. Python standard library
handles extraction and analysis; numpy/matplotlib are used only for figures.

Implementation modules:

- `core.py`: lossless original character coordinates, structural units, and
  separate Explicit/Semantic/Potential candidate rules.
- `events.py`: observed tool/runner events, information replacement candidates,
  and complete-window matched pseudo-event controls.
- `analysis_stats.py`: right censoring, task-cluster summaries, cross-seed
  entity alignment, and coarse two-case KV mapping.
- `prediction.py`: temporal features and task-grouped out-of-fold rule baselines.
- `reporting.py`: examples, plots, Chinese reports, and counterfactual interface.
- `validate_output.py`: independent referential, temporal, and source validation.

The second phase is a design only. Its runner contract requires fresh inference,
environment restoration, independent grading, and measured costs. Missing
high-confidence dependency labels, true decay latency, semantic residual value,
and causal importance remain unidentified rather than being set to zero.
