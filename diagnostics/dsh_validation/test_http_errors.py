#!/usr/bin/env python3
"""Real CPU Chromium regression: HTTP 404 is not a transport failure."""
import argparse
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
(fixture / "index.html").write_text(
    '<!doctype html><meta charset="utf-8"><link rel="icon" href="data:,">'
    '<title>HTTP failure infrastructure fixture</title>'
    '<h1>HTTP failure infrastructure fixture, not a model artifact</h1>'
    '<script>fetch("/absent-script.js").then(async response => {'
    'await response.text(); window.missingScriptStatus = response.status;});</script>\n'
)
plan = output / "plan.json"
plan.write_text(json.dumps({"initial_wait_ms": 300, "actions": []}) + "\n")
command = [
    sys.executable, str(Path(__file__).with_name("run.py")),
    "--root", str(fixture), "--output", str(output / "result"),
    "--plan", str(plan),
]
if args.runtime_root:
    command += ["--runtime-root", args.runtime_root]
if args.tool_root:
    command += ["--tool-root", args.tool_root]
result = subprocess.run(command, text=True, capture_output=True, timeout=120)
(output / "stdout.log").write_text(result.stdout)
(output / "stderr.log").write_text(result.stderr)
report = json.loads((output / "result/report.json").read_text())
assert result.returncode == 1, result.stdout + result.stderr
assert report["status"] == "issues-observed", report
assert report["main_status"] == 200, report
assert report["summary"]["no_failed_requests"] is True, report["request_failures"]
assert report["summary"]["no_http_errors"] is False, report["summary"]
assert any(item["status"] == 404 and item["url"].endswith("/absent-script.js")
           for item in report["http_errors"]), report["http_errors"]
print(json.dumps({
    "passed": True, "http_errors": report["http_errors"],
    "no_failed_requests": report["summary"]["no_failed_requests"],
    "no_http_errors": report["summary"]["no_http_errors"],
    "evidence": str(output),
}, indent=2))
