#!/usr/bin/env python3
"""Prepare isolated marker and CSGO tasks. Does not start DSH or contact any service."""
import argparse
import hashlib
import json
import os
from pathlib import Path
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
    args = parser.parse_args()
    os.umask(0o077)
    source = Path(__file__).resolve().parent
    harness = args.harness.resolve()
    node = harness / "node-v24.19.0-linux-x64/bin/node"
    cli = harness / "node_modules/@deepseek-ai/dsh/lib/bin.js"
    if not node.is_file() or not cli.is_file():
        parser.error("Pinned Node and DSH installation are required")
    if args.context_window < 8192:
        parser.error("context-window must cover the fixed maxTokens=8192 profile")
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    snapshots = root / "configuration"
    snapshots.mkdir()
    names = ["single_trajectory.patch.yml", "provider.patch.yml", "session_runner.patch.yml", "session_runner.mjs"]
    for name in names:
        shutil.copy2(source / name, snapshots / name)
    provider = snapshots / "provider.patch.yml"
    provider.write_text(provider.read_text().replace("contextWindow: 24576", f"contextWindow: {args.context_window}"))
    runner_patch = snapshots / "session_runner.patch.yml"
    runner_patch.write_text(runner_patch.read_text().replace(
        "__DSH_GC_RUNNER__", json.dumps(str(snapshots / "session_runner.mjs"))))
    manifest = {
        "harness": str(harness), "node": str(node), "cli": str(cli),
        "provider": "gc-local", "model": "kvmem-gc-qwen38",
        "base_url": "http://127.0.0.1:18387/v1", "context_window": args.context_window, "max_tokens": 8192,
        "native_k": 4096, "native_g": 4096, "query_replay_budget": 128, "max_inflight": 1,
        "configuration_sha256": {name: hashlib.sha256((snapshots / name).read_bytes()).hexdigest() for name in names},
        "prepared_only": True,
    }
    write_json(root / "manifest.json", manifest)
    for name in ("marker", "csgo"):
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
    print(json.dumps({"run_root": str(root), "prepared_only": True, "model_requests": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
