#!/usr/bin/env python3
"""CPU Chromium infrastructure fixture: form events and same-origin persistence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--runtime-root")
parser.add_argument("--tool-root")
args = parser.parse_args()
output = args.output.resolve()
if output.exists():
    raise SystemExit(f"Refusing to overwrite existing evidence: {output}")
fixture = output / "fixture"
fixture.mkdir(parents=True)
source = fixture / "index.html"
source.write_text("""<!doctype html><meta charset="utf-8"><link rel="icon" href="data:,">
<title>Form observer infrastructure fixture</title>
<h1>Form observer infrastructure fixture, not a model artifact</h1>
<form id="form">
<label>Note <textarea id="note" class="editable">replace this</textarea></label>
<label>Amount <input id="amount" class="editable" type="number" step=".01" value="0"></label>
<label>Category <select id="category"><option value="personal">Personal</option>
<option value="work">Work</option></select></label><button id="save">Save</button>
</form><output id="saved"></output>
<script>
const form = document.querySelector('#form');
const note = document.querySelector('#note');
const amount = document.querySelector('#amount');
const category = document.querySelector('#category');
const saved = document.querySelector('#saved');
function show(record) {
  saved.textContent = record.note + ' / ' + record.amount + ' / ' + record.category;
}
const query = new URLSearchParams(location.search);
const record = JSON.parse(localStorage.getItem('observer-form') || 'null');
if (record && !query.has('forget')) {
  note.value = record.note; amount.value = record.amount; category.value = record.category;
  show(record);
}
form.addEventListener('submit', event => {
  event.preventDefault();
  const record = {note: note.value, amount: amount.value, category: category.value};
  localStorage.setItem('observer-form', JSON.stringify(record)); show(record);
});
const visits = Number(sessionStorage.getItem('visits') || '0') + 1;
sessionStorage.setItem('visits', String(visits));
if (query.has('missing_on_reload') && visits > 1) {
  fetch('/absent-after-reload.js').then(response => response.text());
}
</script>
""")
original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
note = 'Review "quote" — 中文 🧾\nsecond line'
record = dict(note=note, amount="12.50", category="work")
visible = note + " / 12.50 / work"
probe = """({
 note: document.querySelector('#note').value,
 amount: document.querySelector('#amount').value,
 category: document.querySelector('#category').value,
 visible: document.querySelector('#saved').textContent,
 stored: JSON.parse(localStorage.getItem('observer-form') || 'null')
})"""
plan = dict(initial_wait_ms=100, probe_expression=probe, actions=[
    dict(name="fill-note", type="fill", selector="#note", value=note,
         checks=[dict(path="note", op="equals", value=note)]),
    dict(name="fill-amount", type="fill", selector="#amount", value="12.50",
         checks=[dict(path="amount", op="equals", value="12.50")]),
    dict(name="select-category", type="select-option", selector="#category", value="work",
         checks=[dict(path="category", op="equals", value="work")]),
    dict(name="submit", type="click", selector="#save",
         checks=[dict(path="stored", op="equals", value=record),
                 dict(path="visible", op="equals", value=visible)]),
    dict(name="clear-unsaved-input", type="fill", selector="#note", value="",
         checks=[dict(path="note", op="equals", value="")]),
    dict(name="reload", type="reload",
         checks=[dict(path="note", op="equals", value=note),
                 dict(path="amount", op="equals", value="12.50"),
                 dict(path="category", op="equals", value="work"),
                 dict(path="stored", op="equals", value=record),
                 dict(path="visible", op="equals", value=visible)]),
])
cases = {}


def run_case(name, case_plan, entry="index.html"):
    directory = output / name
    directory.mkdir()
    plan_path = directory / "plan.json"
    plan_path.write_text(json.dumps(case_plan, ensure_ascii=False) + "\n")
    command = [sys.executable, str(Path(__file__).with_name("run.py")),
               "--root", str(fixture), "--entry", entry,
               "--output", str(directory / "result"), "--plan", str(plan_path)]
    if args.runtime_root:
        command += ["--runtime-root", args.runtime_root]
    if args.tool_root:
        command += ["--tool-root", args.tool_root]
    result = subprocess.run(command, text=True, capture_output=True, timeout=120)
    (directory / "stdout.log").write_text(result.stdout)
    (directory / "stderr.log").write_text(result.stderr)
    report = json.loads((directory / "result/report.json").read_text())
    assert report["browser"]["rendering"] == "forced-software", report.get("browser")
    renderer = report["browser"]["webgl"]
    assert not renderer["available"] or any(
        item in renderer["renderer"].lower() for item in ("swiftshader", "software", "llvmpipe")
    ), renderer
    cases[name] = dict(exit_code=result.returncode, status=report["status"],
                       summary=report["summary"], report=str(directory / "result/report.json"))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    return result, report


result, good = run_case("persist", plan)
assert result.returncode == 0, result.stdout + result.stderr
assert good["status"] == "observational-checks-passed"
assert good["summary"]["state_checks_passed"] == good["summary"]["state_checks_total"] == 11
assert good["summary"]["action_errors"] == 0
assert good["summary"]["no_http_errors"] is True
assert good["actions"][2]["selected_values"] == ["work"]
assert good["actions"][-1]["reload_status"] == 200
assert good["actions"][-1]["before"]["probe"]["note"] == ""
assert good["actions"][-1]["after"]["probe"]["note"] == note
assert any(e["kind"] == "input" for e in good["actions"][0]["after"]["events"])
assert any(e["kind"] == "change" for e in good["actions"][2]["after"]["events"])
assert good["artifact_sha256"]["index.html"] == original_hash

# LocalStorage still contains the record, but deliberately broken UI restore must fail.
result, broken = run_case("broken-restore", plan, "index.html?forget=1")
assert result.returncode == 1 and broken["status"] == "issues-observed"
assert broken["summary"]["action_errors"] == 0
assert broken["summary"]["no_http_errors"] is True
assert broken["summary"]["specified_state_checks_pass"] is False
assert broken["actions"][-1]["after"]["probe"]["stored"] == record
assert any(not c["pass"] for c in broken["actions"][-1]["checks"])

# New form actions must reject an ambiguous selector, rather than editing the first match.
bad_plan = dict(initial_wait_ms=100, probe_expression=probe, actions=[
    dict(type="fill", selector=".editable", value="wrong-target")])
result, ambiguous = run_case("ambiguous-target", bad_plan)
assert result.returncode == 1 and ambiguous["summary"]["action_errors"] == 1
assert "strict mode" in ambiguous["actions"][0]["error"].lower()
assert ambiguous["actions"][0]["before"]["probe"] == ambiguous["actions"][0]["after"]["probe"]

# A successful reload action must not mask a resource HTTP error from the new document.
reload_plan = dict(initial_wait_ms=100, actions=[dict(type="reload", settle_ms=300)])
result, missing = run_case("reload-http404", reload_plan, "index.html?missing_on_reload=1")
assert result.returncode == 1 and missing["status"] == "issues-observed"
assert missing["summary"]["action_errors"] == 0
assert missing["actions"][0]["reload_status"] == 200
assert missing["summary"]["no_failed_requests"] is True
assert missing["summary"]["no_http_errors"] is False
assert any(item["status"] == 404 and item["url"].endswith("/absent-after-reload.js")
           for item in missing["http_errors"]), missing["http_errors"]

evidence = dict(passed=True, scope="CPU-only infrastructure fixture; no model requests",
                fixture_sha256=original_hash, source_unchanged=True, cases=cases)
(output / "summary.json").write_text(json.dumps(evidence, indent=2) + "\n")
print(json.dumps(evidence, indent=2))
