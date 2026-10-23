#!/usr/bin/env python3
"""Atomically enqueue one human followup or graceful close; no network or model call."""
import argparse
import fcntl
import json
import os
from pathlib import Path


def enqueue(case_dir: Path, *, text: str | None = None, close: bool = False) -> Path:
    commands = case_dir / "commands"
    if not commands.is_dir():
        raise ValueError("Run prepare.py first")
    if not close and (not isinstance(text, str) or not text.strip()):
        raise ValueError("Nonempty task text is required")
    with (case_dir / ".submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        files = sorted(commands.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json"))
        if any(json.loads(p.read_text())["kind"] == "close" for p in files):
            raise ValueError("A close command is already queued")
        index = int(files[-1].stem) + 1 if files else 1
        body = {"index": index, "kind": "close" if close else "user"}
        if not close:
            body["text"] = text
        final = commands / f"{index:06d}.json"
        temp = commands / f".{index:06d}.{os.getpid()}.tmp"
        with temp.open("x") as stream:
            json.dump(body, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task-file", type=Path)
    group.add_argument("--close", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    print(enqueue(args.case_dir.resolve(), text=args.task_file.read_text() if args.task_file else None, close=args.close))


if __name__ == "__main__":
    main()
