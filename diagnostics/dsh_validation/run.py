#!/usr/bin/env python3
"""Launch the independent Chromium observer with isolated CPU dependencies."""
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--tool-root", default="/root/data/kvmem-vllm-gc/dsh-harness/browser-validation")
parser.add_argument("--node")
parser.add_argument("--runtime-root")
known, rest = parser.parse_known_args()
tool_root = Path(known.tool_root).resolve()
runtime_root = Path(known.runtime_root).resolve() if known.runtime_root else tool_root
node = Path(known.node) if known.node else tool_root.parent / "node-v24.19.0-linux-x64/bin/node"
script = Path(__file__).resolve().with_name("validate_static_game.mjs")
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = ""
env["PLAYWRIGHT_BROWSERS_PATH"] = str(runtime_root / "browsers")
libdirs = [runtime_root / "sysroot/usr/lib/x86_64-linux-gnu", runtime_root / "sysroot/lib/x86_64-linux-gnu"]
env["LD_LIBRARY_PATH"] = ":".join([str(p) for p in libdirs if p.exists()] + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
if (runtime_root / "fonts.conf").exists():
    env["FONTCONFIG_FILE"] = str(runtime_root / "fonts.conf")
try:
    os.nice(10)
except OSError:
    pass
os.execve(str(node), [str(node), str(script), "--tool-root", str(tool_root), *rest], env)
