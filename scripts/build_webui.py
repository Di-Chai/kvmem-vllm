#!/usr/bin/env python3
"""Build the vendored KVMem console without fetching any inference backend."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "ui"
CHAT = UI / "chat"
DEFAULT_OUTPUT = ROOT / "src/vllm_kvmem/apps/webui/static"


def run(argv: list[str]) -> None:
    subprocess.run(argv, cwd=CHAT, check=True)


def dependency_notices(output: Path) -> None:
    """Retain package identity and available license text with distributed assets."""
    lock = json.loads((CHAT / "package-lock.json").read_text())
    packages = []
    notices = []
    for relative, metadata in sorted(lock["packages"].items()):
        if not relative:
            continue
        name = relative.rsplit("node_modules/", 1)[-1]
        packages.append({"name": name, "version": metadata.get("version"),
                         "license": metadata.get("license"),
                         "integrity": metadata.get("integrity")})
        package = CHAT / relative
        candidates = sorted(p for p in package.glob("*") if p.is_file()
                            and p.name.lower().startswith(("license", "licence", "copying", "notice")))
        if candidates:
            notices.append(f"\n{'=' * 72}\n{name} {metadata.get('version', '')}\n")
            for path in candidates:
                notices.append(f"\n{path.name}\n{path.read_text(errors='replace')}\n")
    (output / "DEPENDENCIES.json").write_text(json.dumps(packages, indent=2) + "\n")
    (output / "THIRD_PARTY_NOTICES.txt").write_text("".join(notices))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--npm", default="npm", help="npm executable or npm-cli.js path")
    parser.add_argument("--node", default="node")
    parser.add_argument("--skip-install", action="store_true", help="reuse installed lockfile dependencies")
    args = parser.parse_args()
    version = subprocess.check_output([args.node, "-p", "process.versions.node"], text=True).strip()
    if int(version.split(".")[0]) < 22:
        parser.error("Node.js 22 or newer is required to rebuild the UI")
    npm = [args.node, args.npm] if args.npm.endswith(".js") else [args.npm]
    if not args.skip_install:
        run(npm + ["ci", "--ignore-scripts", "--no-audit", "--no-fund"])
    run([args.node, "node_modules/@sveltejs/kit/svelte-kit.js", "sync"])
    run([args.node, "node_modules/svelte-check/bin/svelte-check", "--tsconfig", "./tsconfig.json", "--threshold", "error"])
    run([args.node, "node_modules/vite/bin/vite.js", "build"])
    output = args.output.resolve()
    if output == ROOT or output in ROOT.parents or output == UI or UI in output.parents:
        parser.error("output must not overwrite repository or UI sources")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("monitor", "chat"):
        target = output / name
        if target.is_symlink():
            parser.error(f"output directory must not contain symlinks: {target}")
        if target.exists():
            shutil.rmtree(target)
    shutil.copytree(UI / "monitor", output / "monitor")
    shutil.copytree(CHAT / "dist", output / "chat/view")
    for name in ("llama.cpp-LICENSE", "PROVENANCE.json"):
        shutil.copy2(UI / name, output / name)
    dependency_notices(output)
    files = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(output.rglob("*")) if path.is_file() and path.name != "build-manifest.json"}
    (output / "build-manifest.json").write_text(json.dumps({"schema_version": 1,
        "source": json.loads((UI / "PROVENANCE.json").read_text()),
        "dependency_lock_sha256": hashlib.sha256((CHAT / "package-lock.json").read_bytes()).hexdigest(),
        "files": files}, indent=2) + "\n")
    print(f"Built console: {output}")


if __name__ == "__main__":
    main()
