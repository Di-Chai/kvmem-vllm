import json
from pathlib import Path

import pytest

import grade_full_mab_flash as g


def case(tmp_path, arm="layerwise"):
    source = tmp_path / "question.json"
    info = dict(family="longmemeval", source="longmemeval_s*", raw_question="Question?",
                gold_answer=["Gold"], split="Accurate_Retrieval", dataset_row=0, question_index=0)
    g.save(source, dict(info=info))
    fixture = dict(id="case0", input_tokens=123, source=str(source), source_cache_sha256=g.sha(source),
                   token_sha256="tokens", prompt_sha256="prompt", query_sha256="query")
    record = dict(info, id="case0", status="completed", original_input_intact=True, prompt_tokens=123,
                  token_sha256="tokens", prompt_sha256="prompt", query_sha256="query",
                  raw_completion="<think>private</think> Gold", answer="Gold",
                  actual_query_method="mean_k" if arm=="mean_k" else "mean_k_sequential_layerwise",
                  actual_pressure_method="recency" if arm=="mean_k" else "accumulated_attention_per_layer")
    path = tmp_path / "cases/case0/result.json"
    g.save(path, record)
    return path, fixture, record


@pytest.mark.parametrize("arm", g.ARMS)
def test_bound_final_answer_excludes_history_and_thinking(tmp_path, arm):
    path, fixture, _ = case(tmp_path, arm)
    row, binding = g.bind_record(path, fixture, arm)
    assert row["answer"] == "Gold"
    assert "raw_completion" not in row and "layerwise" not in row
    assert binding["sha256"] == g.sha(path)


@pytest.mark.parametrize("field,value", [("original_input_intact",False),
    ("gold_answer",["changed"]), ("query_sha256","changed"),
    ("answer","changed"), ("actual_pressure_method","fallback")])
def test_changed_inputs_gold_or_execution_fail_before_api(tmp_path, field, value):
    path, fixture, record = case(tmp_path)
    record[field] = value
    g.save(path, record)
    with pytest.raises(AssertionError):
        g.bind_record(path, fixture, "layerwise")


def test_semantic_summary_uses_only_common_cases_and_never_zeros_pending(tmp_path):
    g.save(tmp_path / "judgments/mean_k/a.json",dict(id="a",source="longmemeval_s*",score=True))
    g.save(tmp_path / "judgments/layerwise/a.json",dict(id="a",source="longmemeval_s*",score=False))
    g.save(tmp_path / "judgments/layerwise/b.json",dict(id="b",source="infbench_sum",score=.4))
    report = g.summarize(tmp_path)
    assert report["arms"]["mean_k"]["completed"] == 1
    assert report["arms"]["mean_k"]["expected"] == 400
    assert report["arms"]["mean_k"]["by_source"]["longmemeval_s*"]["score_percent"] == 100
    assert report["paired"] == dict(n=1,layerwise_wins=0,mean_k_wins=1,ties=0)
    assert report["overall_official_macro"] is None


def test_invalid_judge_score_is_not_silently_clamped_or_zeroed(tmp_path):
    g.save(tmp_path / "judgments/mean_k/a.json",dict(id="a",source="infbench_sum",score=1.2))
    with pytest.raises(AssertionError):
        g.summarize(tmp_path)


def test_frozen_evaluator_and_inference_bindings_are_checked(tmp_path):
    runtime = tmp_path / "runtime/script.py"
    runtime.parent.mkdir()
    runtime.write_text("original")
    profile = tmp_path / "inference-profile.json"
    g.save(profile,dict(budget=36864))
    plan = dict(files={"runtime/script.py":g.sha(runtime)},inference_bindings={str(profile):g.sha(profile)})
    g.save(tmp_path / "plan.json",plan)
    g.save(tmp_path / "prepared.json",dict(plan_sha256=g.sha(tmp_path / "plan.json")))
    assert g.verify(tmp_path) == plan
    profile.write_text("{}")
    with pytest.raises(AssertionError):
        g.verify(tmp_path)
