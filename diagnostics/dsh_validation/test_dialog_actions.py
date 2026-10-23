#!/usr/bin/env python3
"""CPU browser fixture: confirmation is explicit, scoped and fail-closed."""
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
<title>Dialog observer infrastructure fixture</title>
<h1>Dialog observer infrastructure fixture, not a model artifact</h1>
<button id="remove">Remove one fixture row</button>
<button id="silent">No dialog</button><button id="wrong">Unexpected alert</button>
<button id="twice">Two confirms</button><output id="rows">3</output>
<script>
document.querySelector('#remove').onclick = () => {
  if (confirm('Delete one fixture row?')) {
    const rows = document.querySelector('#rows');
    rows.textContent = String(Number(rows.textContent) - 1);
  }
};
document.querySelector('#wrong').onclick = () => alert('This is an alert, not a confirm');
document.querySelector('#twice').onclick = () => {
  confirm('First confirmation'); confirm('Unexpected second confirmation');
};
</script>
""")
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
probe = "({rows:Number(document.querySelector('#rows').textContent)})"
cases = {}


def click(selector="#remove", expected=None, checks=None):
    action = dict(type="click", selector=selector,
                  checks=checks or [dict(path="rows", op="equals", value=3)])
    if expected is not None:
        action["expect_dialog"] = expected
    return action


def run_case(name, actions):
    directory = output / name
    directory.mkdir()
    plan = directory / "plan.json"
    plan.write_text(json.dumps(dict(initial_wait_ms=100, probe_expression=probe,
                                   actions=actions)) + "\n")
    command = [sys.executable, str(Path(__file__).with_name("run.py")),
               "--root", str(fixture), "--output", str(directory / "result"),
               "--plan", str(plan)]
    if args.runtime_root:
        command += ["--runtime-root", args.runtime_root]
    if args.tool_root:
        command += ["--tool-root", args.tool_root]
    result = subprocess.run(command, text=True, capture_output=True, timeout=120)
    (directory / "stdout.log").write_text(result.stdout)
    (directory / "stderr.log").write_text(result.stderr)
    report = json.loads((directory / "result/report.json").read_text())
    assert report["summary"]["no_http_errors"] is True
    assert report["summary"]["no_console_errors"] is True
    assert "--disable-gpu" in report["browser"]["actual_command_line"]
    webgl = report["browser"]["webgl"]
    assert not webgl["available"] or any(s in webgl["renderer"].lower()
        for s in ("swiftshader", "software", "llvmpipe")), webgl
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    cases[name] = dict(exit_code=result.returncode, status=report["status"],
                       summary=report["summary"], report=str(directory / "result/report.json"))
    return result, report


accept = dict(type="confirm", accept=True)
dismiss = dict(type="confirm", accept=False)
result, good = run_case("explicit-and-default", [
    click(), click(expected=dismiss),
    click(expected=accept, checks=[dict(path="rows", op="decreased"),
                                  dict(path="rows", op="equals", value=2)]),
    click(checks=[dict(path="rows", op="equals", value=2)]),
])
assert result.returncode == 0 and good["summary"]["state_checks_passed"] == 5
assert "dialogs" not in good["actions"][0] and "dialogs" not in good["actions"][3]
for index, decision in [(1, "dismiss"), (2, "accept")]:
    dialog, = good["actions"][index]["dialogs"]
    assert dialog == dict(type="confirm", message="Delete one fixture row?",
                         decision=decision, matches_expected_type=True, handled=True)

for name, selector, expected, message, count in [
    ("missing", "#silent", accept, "not observed within 3000 ms", 0),
    ("wrong-type", "#wrong", accept, "observed alert", 1),
    ("duplicate", "#twice", accept, "observed 2", 2),
    ("invalid-decision", "#remove", dict(type="confirm", accept="true"), "accept:boolean", 0),
]:
    # A following ordinary click proves the per-action listener was removed on failure.
    result, bad = run_case(name, [click(selector, expected), click()])
    assert result.returncode == 1 and bad["status"] == "issues-observed"
    assert bad["summary"]["action_errors"] == 1
    assert bad["summary"]["state_checks_passed"] == 2
    action = bad["actions"][0]
    assert message in action["error"], action
    assert len(action.get("dialogs", [])) == count
    assert "dialogs" not in bad["actions"][1]
    if name == "wrong-type":
        assert action["dialogs"][0]["type"] == "alert"
        assert action["dialogs"][0]["decision"] == "dismiss"
    if name == "duplicate":
        assert [d["decision"] for d in action["dialogs"]] == ["accept", "dismiss"]

evidence = dict(passed=True, scope="CPU-only infrastructure fixture; no model requests",
                fixture_sha256=source_hash, source_unchanged=True, cases=cases)
(output / "summary.json").write_text(json.dumps(evidence, indent=2) + "\n")
print(json.dumps(evidence, indent=2))
