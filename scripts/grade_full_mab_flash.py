"""Grade this complete-MAB comparison with the established Flash rubrics.

Independent of inference: immutable completed-answer batches, four API callers,
the existing official-prompt adapter, and no model-answer retries.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from types import SimpleNamespace as NS

REPO = Path(__file__).resolve().parents[1]
ARMS = ("mean_k", "layerwise")
OFFICIAL = Path("/home/chaidi/MemoryAgentBench-official")
LEGACY = REPO / "artifacts/flash-regrade-20261021-r1/evaluation/evaluators"
PINNED = "455306dcabc3842526eb83cd4e225e5d486c5c5d"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(campaign, output):
    assert not (output / "plan.json").exists(), "Use a fresh grading root"
    expected = [ident for group in read(campaign / "data/cohort.json")["groups"]
                if group["source"].startswith(("longmemeval_", "infbench_"))
                for ident in group["cases"]]
    assert len(expected) == len(set(expected)) == 400
    runtime = output / "runtime"
    runtime.mkdir()
    for source, name in ((Path(__file__), "grade_full_mab_flash.py"),
                         (REPO / "scripts/regrade_flash.py", "regrade_flash.py"),
                         (LEGACY / "historical_lme_score.py", "transport.py"),
                         (LEGACY / "judge_memoryagentbench_special_flash.py", "special.py"),
                         (LEGACY / "historical_lme_calibration.json", "calibration.json")):
        shutil.copy2(source, runtime / name)
    for name in ("longmem_qa_evaluate.py", "summarization_evaluate.py"):
        shutil.copy2(campaign / "evaluation/llm_based_eval" / name, runtime / name)
    bindings = {}
    for arm in ARMS:
        for name in ("fixture.json", "profile.json"):
            path = campaign / f"formal_{arm}" / name
            bindings[str(path)] = sha(path)
    save(output / "plan.json", dict(campaign=str(campaign), expected_semantic_per_arm=400,
        expected_ids=expected, arms=list(ARMS), workers=4, official_commit=PINNED,
        model="deepseek-flash", endpoint="https://api.deepseek.com", external_api=True,
        authorization="User explicitly requested DS 4.1 Flash grading for the current two-method full MAB comparison, including its subsequent answers (2026-11-07).",
        transmitted="Question, reference/rubric/keypoints, and final model answer; method identity and long input histories are excluded.",
        lme=dict(thinking="enabled", max_tokens=1024, temperature=0),
        summary=dict(thinking="disabled", max_tokens=4096, temperature=0),
        retries="Established transport retries technical failures; length-only 4x cap to 16384; one versioned JSON-format repair. Never retry a valid score.",
        local_metrics="Unchanged official deterministic metrics; format-extraction diagnostics remain separate.",
        limitation="Official rubrics with a DeepSeek judge, not the paper's GPT-4 judge. No overall macro is inferred from semantic-only scores.",
        files={str(p.relative_to(output)):sha(p) for p in runtime.iterdir() if p.is_file()},
        inference_bindings=bindings, supersedes_assistant_judge_only=True))
    save(output / "prepared.json", dict(plan_sha256=sha(output / "plan.json")))
    save(output / "status.json", dict(status="prepared", expected=800, at=time.time()))


def verify(output):
    assert sha(output / "plan.json") == read(output / "prepared.json")["plan_sha256"]
    plan = read(output / "plan.json")
    for name, expected in plan["files"].items():
        assert sha(output / name) == expected, name
    for name, expected in plan["inference_bindings"].items():
        assert sha(Path(name)) == expected, name
    return plan


def bind_record(path, fixture_row, arm):
    raw = path.read_bytes()
    row = json.loads(raw)
    assert row["id"] == fixture_row["id"] == path.parent.name
    assert row["status"] == "completed" and row["original_input_intact"] is True
    assert row["prompt_tokens"] == fixture_row["input_tokens"]
    for field in ("token_sha256", "prompt_sha256", "query_sha256"):
        assert row[field] == fixture_row[field], field
    assert row["answer"] == row["raw_completion"].split("</think>", 1)[-1].strip()
    assert row["actual_query_method"] == ("mean_k" if arm == "mean_k" else "mean_k_sequential_layerwise")
    assert row["actual_pressure_method"] == ("recency" if arm == "mean_k" else "accumulated_attention_per_layer")
    source = Path(fixture_row["source"])
    assert sha(source) == fixture_row["source_cache_sha256"]
    info = read(source)["info"]
    assert all(row[k] == v for k, v in info.items()), "Question/reference binding changed"
    assert row["family"] in ("longmemeval", "infbench")
    compact = {k:row.get(k) for k in ("id", "split", "dataset_row", "question_index", "source", "family", "gold_answer", "question_type", "qa_pair_id", "keypoints", "raw_question", "answer")}
    binding = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())
    return compact, binding


def summarize(output, expected=400):
    arms, values = {}, {}
    for arm in ARMS:
        rows = [read(p) for p in (output / "judgments" / arm).glob("*.json")]
        assert len({r["id"] for r in rows}) == len(rows)
        groups = defaultdict(list)
        values[arm] = {}
        for row in rows:
            score = float(row["score"])
            assert 0 <= score <= 1, "Invalid judge score; preserve and inspect"
            values[arm][row["id"]] = score
            groups[row["source"]].append(score)
        arms[arm] = dict(completed=len(rows), expected=expected,
            by_source={key:dict(n=len(vals), score_percent=100*sum(vals)/len(vals)) for key, vals in groups.items()})
    common = values["mean_k"].keys() & values["layerwise"].keys()
    delta = [values["layerwise"][key]-values["mean_k"][key] for key in common]
    return dict(judge="deepseek-flash", arms=arms,
        paired=dict(n=len(common), layerwise_wins=sum(d>1e-12 for d in delta), mean_k_wins=sum(d < -1e-12 for d in delta), ties=sum(abs(d)<=1e-12 for d in delta)),
        overall_official_macro=None, at=time.time())


def run(output, credential):
    assert Path(__file__).resolve() == output / "runtime/grade_full_mab_flash.py", "Use frozen grader"
    plan = verify(output)
    campaign = Path(plan["campaign"])
    sys.path.insert(0, str(output / "runtime"))
    import regrade_flash as f
    transport = load(output / "runtime/transport.py", "full_mab_flash_transport")
    assert transport.API_BASE == plan["endpoint"]
    flash = f.Flash(output, transport, credential.read_text().strip())
    models, _ = flash.client.request("/models")
    assert f.MODEL in {item["id"] for item in models["data"]}
    fixtures = {arm:{r["id"]:r for r in read(campaign / f"formal_{arm}/fixture.json")["rows"]} for arm in ARMS}
    expected = set(plan["expected_ids"])
    assert all(expected <= rows.keys() for rows in fixtures.values())
    guard = threading.Lock()
    call_locks = {}
    def call(session, args, api_key, prompt, max_tokens=512, thinking="enabled"):
        body = dict(model=f.MODEL, messages=[dict(role="user", content=prompt)], temperature=0,
                    max_tokens=max_tokens, stream=False, thinking=dict(type=thinking))
        with guard:
            lock = call_locks.setdefault(f.digest(body), threading.Lock())
        with lock:
            return f.complete_mab_call(flash, body)
    special = load(output / "runtime/special.py", "full_mab_calibration")
    calibration = output / "calibration.json"
    if not calibration.exists():
        def calibrate(item):
            ref = item["fixture"]
            prompt = special.longmem_prompt(ref["question_type"], ref["question"], ref["answer"], item["hypothesis"], ref["question_id"].endswith("_abs"))
            text = call(None, None, None, prompt, max_tokens=1024)
            return dict(expected=item["expected"], observed=("yes" in text.lower()), output=text)
        with ThreadPoolExecutor(max_workers=4) as pool:
            checks = list(pool.map(calibrate, read(output / "runtime/calibration.json")["cases"]))
        save(calibration, dict(passed=all(x["expected"]==x["observed"] for x in checks), cases=checks))
    assert read(calibration)["passed"], "Judge calibration failed; do not resample"
    os.environ["DEEPSEEK_API_KEY"] = "transport-managed"
    sequence = len(list((output / "batches").glob("*")))
    def grade_shard(job):
        arm, shard = job
        module = load(output / "runtime/special.py", "mab_shard_"+shard.name+arm)
        args = NS(results_dir=shard, official_repo=OFFICIAL, model=f.MODEL,
            base_url=plan["endpoint"], thinking="enabled", infbench_thinking="disabled",
            longmem_max_tokens=1024, allow_partial=True, limit=None, timeout=60, max_retries=4)
        module.parse_args = lambda:args
        module.call = call
        original_load = module.load_string_constants
        module.load_string_constants = lambda path,names:original_load(output / "runtime/summarization_evaluate.py", names)
        proof = read(shard / "snapshot.json")
        assert sha(shard / "rows/all/results.jsonl") == proof["rows_sha256"]
        assert sha(shard / "bindings.json") == proof["bindings_sha256"]
        assert module.main() == 0
        assert sha(shard / "rows/all/results.jsonl") == proof["rows_sha256"]
        records = [json.loads(line) for line in (shard / "rows/all/results.jsonl").read_text().splitlines()]
        mapping = {module.key(row):row for row in records}
        judged = [json.loads(line) for line in (shard / "special_judgments.jsonl").read_text().splitlines()]
        assert len(judged) == len(mapping) == len(records)
        for row in judged:
            ident = mapping[row["key"]]["id"]
            binding = read(shard / "bindings.json")[ident]
            assert sha(Path(binding["path"])) == binding["sha256"]
            assert 0 <= float(row["score"]) <= 1, "Judge score out of range"
            destination = output / "judgments" / arm / (ident+".json")
            assert not destination.exists(), "Never replace completed judgments"
            save(destination, dict(row, id=ident, source_binding=binding, shard=str(shard), response_model=flash.returned_model))
    with (output / "grading.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(output / "process.json", dict(pid=os.getpid(), at=time.time()))
        while True:
            verify(output)
            jobs = []
            for arm in ARMS:
                todo = [ident for ident in sorted(expected) if not (output / "judgments" / arm / (ident+".json")).exists()
                    and (campaign / f"formal_{arm}/run/answers/cases" / ident / "result.json").exists()]
                if not todo:
                    continue
                sequence += 1
                partitions = [todo[index:index+16] for index in range(0, len(todo), 16)]
                for index, identities in enumerate(partitions):
                    if not identities:
                        continue
                    shard = output / "batches" / f"{sequence:04d}" / arm / f"shard{index}"
                    dest = shard / "rows/all/results.jsonl"
                    dest.parent.mkdir(parents=True)
                    bindings = {}
                    with dest.open("x") as stream:
                        for ident in identities:
                            path = campaign / f"formal_{arm}/run/answers/cases" / ident / "result.json"
                            compact, binding = bind_record(path, fixtures[arm][ident], arm)
                            bindings[ident] = binding
                            stream.write(json.dumps(compact, ensure_ascii=False)+"\n")
                    save(shard / "bindings.json", bindings)
                    save(shard / "snapshot.json", dict(rows_sha256=sha(dest), bindings_sha256=sha(shard / "bindings.json"), at=time.time()))
                    jobs.append((arm, shard))
            report = summarize(output)
            total = sum(a["completed"] for a in report["arms"].values())
            save(output / "comparison.json", report)
            save(output / "status.json", dict(status="grading" if jobs else "waiting_inference", completed=total, expected=800, response_model=flash.returned_model, at=time.time()))
            if jobs:
                jobs.sort(key=lambda x:hashlib.sha256(str(x[1].relative_to(output)).encode()).hexdigest())
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [pool.submit(grade_shard, job) for job in jobs]
                    for future in as_completed(futures):
                        future.result()
                        report = summarize(output)
                        save(output / "comparison.json", report)
                        save(output / "status.json", dict(status="grading", completed=sum(a["completed"] for a in report["arms"].values()), expected=800, response_model=flash.returned_model, at=time.time()))
                continue
            if total == 800:
                save(output / "status.json", dict(status="completed", completed=total, expected=800, response_model=flash.returned_model, at=time.time()))
                break
            inference = read(campaign / "status.json")
            if inference["status"] in ("failed", "paused", "stopped", "inference_completed"):
                raise RuntimeError("Missing semantic answers while inference is "+inference["status"])
            time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--credential-file", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "prepare":
        prepare(args.campaign.resolve(), output)
    else:
        try:
            run(output, args.credential_file)
        except BaseException as error:
            save(output / "status.json", dict(status="failed", error=repr(error), at=time.time()))
            raise
