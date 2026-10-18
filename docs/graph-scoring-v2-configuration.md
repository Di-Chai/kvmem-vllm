# Graph scoring V2 configuration

Graph scoring remains opt-in: the default mode is `off`, and an omitted
`graph_scoring_version` means `v1`. Existing V1 dual-weight profiles retain their
interpretation. The Mean-K score-domain correction is a separate baseline change;
selecting V1 or disabling graph propagation does not restore the old mandatory
softmax denominator.

The complete profiles are
[`graph-scoring-v2-direct.json`](../examples/graph-scoring-v2-direct.json) and
[`graph-scoring-v2-elastic.json`](../examples/graph-scoring-v2-elastic.json).
They preserve the existing graph example's model, K/G budgets, batch budget,
sampling and resource settings. Configure the local model/tokenizer paths and,
for the elastic gateway, the existing `users_file` credential file before use.
These are configurations, not model-quality qualification results.

## Canonical fields

Place these fields under `kvmem` for direct serving, or at the top level for
elastic serving:

```json
{
  "graph_mode": "multi",
  "graph_scoring_version": "v2",
  "graph_alpha": 0.2,
  "pressure_policy": "graph",
  "action_attention": true,
  "prefill_attention": true,
  "action_scope": "non-thinking",
  "action_fraction": 0.0,
  "query_history": false,
  "repository_gc": false,
  "request_planning": true
}
```

`graph_alpha` is the shared Query/pressure propagation coefficient. It defaults
to `0.2`, must be a finite number in `[0,1]`, and rejects booleans. V2 accepts no
explicit `graph_locality_weight` or `graph_attention_weight`, including their old
default values. There is no conversion from the V1 pair to alpha. An explicitly
supplied alpha requires V2, so a V1 profile cannot silently ignore it.

| Pressure policy | Behavior | Legacy alias when explicitly present |
| --- | --- | --- |
| `recency` | Existing temporal pressure selection | `action_pressure=false` |
| `attention` | Existing cumulative-attention pressure selection | `action_pressure=true` |
| `graph` | One-hop graph propagation of cumulative-attention seeds, followed by the pressure picker | No legacy boolean alias |

If the enum is omitted, the old boolean resolves to `recency` or `attention`.
If both are explicitly supplied, only the equivalent aliases in the table are
accepted. Any explicit boolean with `graph` is rejected. Internally, the frozen
runtime options expose the resolved enum and a compatibility `action_pressure`
boolean that is true for both attention-based policies. Selection dispatch must
use the enum.

Graph pressure requires V2 and `graph_mode=locality`, `attention`, or `multi`.
It also requires output/prefill attention collection and
`action_scope=non-thinking`; there is no implicit scoring mode for `off` or
`collect`. V2 Query scoring can be paired with the existing `attention` policy
for controlled comparison. The legacy V1 scoring restrictions remain intact.

## Serialization and sessions

The launchers distinguish explicitly supplied fields from injected defaults.
V2 resolved profiles and worker connector payloads contain version, alpha and
the resolved pressure policy, and omit both legacy graph weights and
`action_pressure`. Reloading their JSON therefore cannot turn old defaults into
conflicting user settings. V1 serialization omits inactive alpha and retains
its legacy weight interpretation. The elastic launcher also accepts its own
resolved launch configuration, rebuilding derived paths and backend settings.

The worker's frozen runtime options own the scoring strategy for the session
lifetime. These settings are not per-request planning fields. Reusing a session
must not be used to switch scoring version, alpha or pressure policy.

The elastic append protocol forwards `kvmem_locality_spans` and
`kvmem_locality_provenance` through the same request-plan allowlist as direct
serving. Span coordinates are absolute offsets in the complete session history,
including previously accepted assistant tokens; they are not suffix-relative
append offsets. The gateway validates them against the full prompt and does not
add an offset. These graph-only fields leave mandatory and Query/replay policy
unchanged.

## Validation

`tests/test_graph_v2_options.py` covers explicit/default conflicts, version and
alpha validation, pressure aliases, graph-pressure dependencies and direct /
elastic / worker JSON round trips. `tests/test_graph_v2_gateway.py` exercises two
real ASGI turns with absolute spans and provenance, verifies token and planning
invariance, and checks that an invalid append preserves history and credits.
These are CPU protocol checks; they do not establish model answer quality or
GPU performance.
