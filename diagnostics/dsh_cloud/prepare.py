#!/usr/bin/env python3
"""Prepare isolated marker, CSGO and optional custom tasks without contacting a service."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil

from submit import enqueue


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--harness", type=Path, default=Path("/root/data/kvmem-vllm-gc/dsh-harness"))
    parser.add_argument("--context-window", type=int, default=24576,
                        help="Must match the already validated backend max_model_len")
    parser.add_argument("--select-budget", type=int, default=4096,
                        help="Record the backend K budget; positive multiple of 32, with K + fixed G=4096 <= context")
    parser.add_argument("--max-output-tokens", type=int, default=8192,
                        help="Provider output cap; must fit context and the separately configured bridge segment limit")
    parser.add_argument("--case-name", help="Additional independent queued-session case, e.g. expense-tracker")
    parser.add_argument("--task-file", type=Path, help="UTF-8 initial task for --case-name; copied into the fresh run")
    args = parser.parse_args()
    if (args.case_name is None) != (args.task_file is None):
        parser.error("--case-name and --task-file must be provided together")
    custom_task = None
    if args.case_name is not None:
        if (not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", args.case_name)
                or args.case_name in ("marker", "csgo", "configuration")):
            parser.error("case-name must be a new lowercase name (letters, digits, _ or -; at most 64 characters)")
        try:
            custom_task = args.task_file.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            parser.error(f"Cannot read custom task: {error}")
        if not custom_task.strip():
            parser.error("Custom task must be nonempty")
    os.umask(0o077)
    source = Path(__file__).resolve().parent
    harness = args.harness.resolve()
    node = harness / "node-v24.19.0-linux-x64/bin/node"
    cli = harness / "node_modules/@deepseek-ai/dsh/lib/bin.js"
    if not node.is_file() or not cli.is_file():
        parser.error("Pinned Node and DSH installation are required")
    if args.context_window < 8192:
        parser.error("context-window must be at least 8192 for this native K/G profile")
    if args.select_budget <= 0 or args.select_budget % 32:
        parser.error("select-budget must be positive and aligned to 32 tokens")
    if args.select_budget + 4096 > args.context_window:
        parser.error("select-budget plus fixed native G=4096 must not exceed context-window")
    if not 1 <= args.max_output_tokens <= args.context_window:
        parser.error("max-output-tokens must be positive and no larger than context-window")
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    snapshots = root / "configuration"
    snapshots.mkdir()
    names = ["single_trajectory.patch.yml", "provider.patch.yml", "session_runner.patch.yml", "session_runner.mjs"]
    for name in names:
        shutil.copy2(source / name, snapshots / name)
    provider = snapshots / "provider.patch.yml"
    provider.write_text(provider.read_text().replace("contextWindow: 24576", f"contextWindow: {args.context_window}")
                        .replace("maxTokens: 8192", f"maxTokens: {args.max_output_tokens}"))
    runner_patch = snapshots / "session_runner.patch.yml"
    runner_patch.write_text(runner_patch.read_text().replace(
        "__DSH_GC_RUNNER__", json.dumps(str(snapshots / "session_runner.mjs"))))
    manifest = {
        "harness": str(harness), "node": str(node), "cli": str(cli),
        "provider": "gc-local", "model": "kvmem-gc-qwen38",
        "base_url": "http://127.0.0.1:18387/v1", "context_window": args.context_window, "max_tokens": args.max_output_tokens,
        "native_k": args.select_budget, "native_g": 4096, "query_replay_budget": 128, "max_inflight": 1,
        "configuration_sha256": {name: hashlib.sha256((snapshots / name).read_bytes()).hexdigest() for name in names},
        "prepared_only": True,
    }
    case_names = ["marker", "csgo"] + ([args.case_name] if args.case_name else [])
    for name in case_names:
        case = root / name
        for directory in ("workspace", "commands", "turns", "home/profiles/headless"):
            (case / directory).mkdir(parents=True)
        profile = case / "home/profiles/headless"
        write_json(profile / "package.json", {
            "name": f"gc-{name}-headless", "private": True, "dependencies": {},
            "dsh": {"profile": {"bundles": ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-headless"]}},
        })
        (profile / "cordis.yml").write_text("[]\n")
        (profile / "cordis.patch.yml").write_text("[]\n")
        (profile / "pnpm-workspace.yaml").write_text("packages:\n  - .\n\nnodeLinker: hoisted\nautoInstallPeers: false\n")
    marker = "GC_MARKER_" + secrets.token_hex(16)
    (root / "marker/workspace/marker.txt").write_text(marker + "\n")
    write_json(root / "marker/expected.json", {"marker": marker, "required_tool": "read", "path": "marker.txt"})
    (root / "marker/task.txt").write_text("请使用 read 工具读取当前工作目录中的 marker.txt。最终回复只输出文件中的完整一行，不添加说明，不修改任何文件。\n")
    # Keep the original user request, without injecting unrelated workload requirements.
    csgo_task = "做个网页版的csgo。"
    (root / "csgo/task.txt").write_text(csgo_task + "\n")
    enqueue(root / "csgo", text=csgo_task)
    if custom_task is not None:
        (root / args.case_name / "task.txt").write_text(custom_task, encoding="utf-8")
        enqueue(root / args.case_name, text=custom_task.strip())
    manifest["cases"] = {}
    for name in case_names:
        case = root / name
        entry = {
            "runner": "headless" if name == "marker" else "session",
            "task_sha256": hashlib.sha256((case / "task.txt").read_bytes()).hexdigest(),
        }
        if name != "marker":
            entry["initial_command_sha256"] = hashlib.sha256(
                (case / "commands/000001.json").read_bytes()).hexdigest()
        manifest["cases"][name] = entry
    write_json(root / "manifest.json", manifest)
    print(json.dumps({"run_root": str(root), "prepared_only": True, "model_requests": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
