#!/usr/bin/env python3
"""Launch one prepared DSH task. This is the explicit model-request boundary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--case", choices=("marker", "csgo"), required=True)
    parser.add_argument("--execute", action="store_true", help="Actually start DSH; default only prints the launch plan")
    args = parser.parse_args()
    os.umask(0o077)
    root = args.run_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    case = root / args.case
    configuration = root / "configuration"
    for name, digest in manifest["configuration_sha256"].items():
        if hashlib.sha256((configuration / name).read_bytes()).hexdigest() != digest:
            parser.error(f"Prepared configuration changed: {name}; prepare a fresh run")
    env = dict(os.environ)
    # Remove task-relevant inherited switches and credential routes from this isolated launch.
    for key in list(env):
        if key.startswith("DSH_") or key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
            env.pop(key)
    env.update({"DSH_HOME": str(case / "home"), "DSH_GC_RUN_DIR": str(case),
                "DSH_GC_HARNESS": manifest["harness"], "DSH_GC_ALLOW_MODEL_REQUESTS": "1",
                "DSH_GC_RUNNER": str(configuration / "session_runner.mjs")})
    env["PATH"] = str(Path(manifest["node"]).parent) + os.pathsep + env.get("PATH", "")
    command = [manifest["node"], manifest["cli"], "--profile", "headless",
               "--patch", str(configuration / "provider.patch.yml"),
               "--patch", str(configuration / "single_trajectory.patch.yml")]
    if args.case == "csgo":
        command.extend(["--patch", str(configuration / "session_runner.patch.yml")])
    command.append((case / "task.txt").read_text().strip())
    plan = {"command": command, "cwd": str(case / "workspace"), "dsh_home": env["DSH_HOME"],
            "execute": args.execute, "provider": manifest["base_url"], "case": args.case}
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return
    # The same run is never implicitly restarted: CSGO continuation uses submit.py.
    with (case / "launch.json").open("x") as stream:
        json.dump(plan | {"started_unix": time.time()}, stream, ensure_ascii=False, indent=2)
    with (case / "stdout.log").open("x") as out, (case / "stderr.log").open("x") as err:
        child = subprocess.Popen(command, cwd=case / "workspace", env=env, stdout=out, stderr=err)
        (case / "pid.json").write_text(json.dumps({"pid": child.pid, "launcher_pid": os.getpid()}) + "\n")
        code = child.wait()
    (case / "exit.json").write_text(json.dumps({"exit_code": code, "finished_unix": time.time()}) + "\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
