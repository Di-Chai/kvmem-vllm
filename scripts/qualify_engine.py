"""Local GPU prerequisite smoke; deliberately not a KVMem throughput benchmark."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request


def fetch(url, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def validate_archive(report, archived):
    if archived["active_requests"] or archived["archive_used_slots"] or archived["pending_downloads"]:
        raise RuntimeError("Finalized requests retained authority resources")
    if not all(r["sealed_records_verified"] and r["history_end"] == r["accepted_tokens"]
               for r in archived["completed"]):
        raise RuntimeError("Incomplete archive verification or accepted-history mismatch")
    items = [report["single"], *report["concurrent"]]
    if len(items) != len(archived["completed"]):
        raise RuntimeError("Native and HTTP request sets differ")
    bound = archived["max_concurrent_batches"] * archived["decode_query_len"]
    for item in items:
        response = item["result"]
        matched = [record for record in archived["completed"]
                   if record["request_id"].startswith(response["id"] + "-0-")]
        if len(matched) != 1:
            raise RuntimeError("Missing or ambiguous request archive")
        record = matched[0]
        expected = response["usage"]["prompt_tokens"] + response["usage"]["completion_tokens"] - 1
        if not expected <= record["history_end"] <= expected + bound:
            raise RuntimeError("Accepted archive extent differs from native request work")
        # Retain native evidence of the last committed packet, rather than
        # silently widening a total-count check to make it pass.
        step = record["last_steps"][-1]
        if step["history_begin"] + step["accepted"] != record["history_end"]:
            raise RuntimeError("Final native capture packet disagrees with archive extent")
    if report["dflash"] and not sum(item["rejected_tokens"] for item in archived["completed"]):
        raise RuntimeError("DFlash qualification did not exercise any rejected candidates")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dflash", action="store_true")
    parser.add_argument("--capture-probe", action="store_true")
    parser.add_argument("--rebuild-probe", action="store_true")
    parser.add_argument("--scheduled-rebuild-probe", action="store_true")
    parser.add_argument("--archive-probe", action="store_true")
    parser.add_argument("--replay-probe", action="store_true")
    parser.add_argument("--compression-probe", action="store_true",
                        help="Single long-request K131072/G65536 selection/replay diagnostic")
    parser.add_argument("--history-repeat", type=int, default=0,
                        help="Prepend deterministic repository-like history to exercise chunked prefill")
    args = parser.parse_args()
    if args.compression_probe:
        if args.dflash or args.replay_probe or args.rebuild_probe or args.scheduled_rebuild_probe or args.history_repeat:
            parser.error("Compression diagnostic requires its own synchronous target-only long workload")
        args.archive_probe = True
    if not 0 <= args.history_repeat <= 128:
        parser.error("history-repeat must be in [0, 128] for this 8192-context diagnostic")
    if args.rebuild_probe or args.scheduled_rebuild_probe or args.archive_probe or args.replay_probe:
        args.capture_probe = True
    if args.archive_probe and (args.rebuild_probe or args.scheduled_rebuild_probe):
        parser.error("Continuous archive qualification uses its own lifecycle diagnostic")
    if args.scheduled_rebuild_probe and (args.dflash or args.rebuild_probe):
        parser.error("Scheduled identity probe currently requires target-only execution")
    args.output.mkdir(parents=True, exist_ok=False)
    models = args.models.resolve()
    scope = ("scheduler/worker identity-refresh diagnostic; no compressed KVMem serving"
             if args.scheduled_rebuild_probe else
             "native identity-refresh diagnostic; no compressed KVMem serving"
             if args.rebuild_probe else "engine prerequisite, no KVMem enabled")
    report = {"scope": scope, "dflash": args.dflash}
    root = Path(__file__).resolve().parent.parent
    sources = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                                      cwd=root, text=True).splitlines()
    manifest = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in sorted(set(sources)) if (root / name).is_file()}
    (args.output / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    report["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    report["source_manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if args.archive_probe:
        report["scope"] = "continuous target KV authority; no compressed KVMem serving"
    with open("/tmp/kvmem-vllm-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        processes = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if processes:
            raise RuntimeError(f"GPU has active compute processes: {processes}")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE="1", VLLM_USE_V2_MODEL_RUNNER="1", MAX_JOBS="2",
                   CUDA_HOME="/usr/local/cuda-13", VLLM_PLUGINS="gguf,kvmem")
        env["PATH"] = f"{Path(sys.executable).parent}:/usr/local/cuda-13/bin:" + env["PATH"]
        if args.capture_probe:
            env["KVMEM_RUNNER_PROBE_DIR"] = str((args.output / "capture").resolve())
        if args.rebuild_probe:
            env["KVMEM_REBUILD_PROBE"] = "1"
        if args.archive_probe:
            env["KVMEM_ARCHIVE_PROBE"] = "1"
        if args.replay_probe:
            env["KVMEM_REPLAY_PROBE"] = "1"
        if args.compression_probe:
            env["KVMEM_COMPRESSION_PROBE"] = "1"
            env["KVMEM_WINDOW_METADATA"] = "1"
        if args.scheduled_rebuild_probe:
            env["KVMEM_SCHEDULED_REBUILD_PROBE"] = "1"
            env["KVMEM_WINDOW_METADATA"] = "1"
        command = [sys.executable, "-m", "vllm_kvmem", "--draft-load-compat", "vllm", "serve",
                   str(models / "Qwen3.8-27B-Q8_0.gguf"),
                   "--served-model-name", "qualification", "--host", "127.0.0.1", "--port", str(port),
                   "--tokenizer", str(models / "Qwen3.8-27B-BF16"),
                   "--hf-config-path", str(models / "Qwen3.8-27B-BF16"),
                   "--config-format", "hf", "--dtype", "bfloat16", "--language-model-only",
                   "--model-loader-extra-config", json.dumps({"mm_proj": str(models / "Qwen3.8-27B-mmproj-f16.gguf")}),
                   "--kv-cache-dtype", "fp8_e4m3", "--max-model-len", "262144" if args.compression_probe else "8192",
                   "--max-num-seqs", "4", "--max-num-batched-tokens", "1024",
                   "--gpu-memory-utilization", "0.72", "--no-enable-prefix-caching",
                   "--generation-config", "vllm", "--enforce-eager"]
        if args.compression_probe:
            command += ["--no-async-scheduling"]
            report["scope"] = "real K128K/G64K target compression/replay; unbounded native allocation; long C1 only"
        if args.scheduled_rebuild_probe:
            command += ["--no-async-scheduling", "--scheduler-cls", "vllm_kvmem.scheduler.KVMemScheduler",
                        "--kv-transfer-config", json.dumps({
                            "kv_connector": "KVMemConnector", "kv_role": "kv_both",
                            "kv_connector_module_path": "vllm_kvmem.connector"})]
        if args.dflash:
            command += ["--speculative-config", json.dumps({
                "method": "dflash", "model": str(models / "Qwen3.8-27B-DFlash2-BF16"),
                "num_speculative_tokens": 7, "draft_load_config": {"load_format": "safetensors"},
                "kv_cache_dtype": "bfloat16"})]
        report["command"] = command
        (args.output / "command.json").write_text(json.dumps(report, indent=2))
        process = None
        try:
            with (args.output / "server.log").open("w") as logfile:
                process = subprocess.Popen(command, env=env, stdout=logfile, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                report["server_pid"] = process.pid
                start = time.monotonic()
                url = f"http://127.0.0.1:{port}"
                while time.monotonic() - start < 600:
                    if process.poll() is not None:
                        raise RuntimeError(f"Server exited {process.returncode}; see server.log")
                    try:
                        fetch(url + "/health")
                        break
                    except json.JSONDecodeError:
                        break  # Health is HTTP 200 with empty body.
                    except (OSError, TimeoutError):
                        time.sleep(1)
                else:
                    raise TimeoutError("Server readiness exceeded 600 seconds")
                report["startup_seconds"] = time.monotonic() - start
                if args.capture_probe:
                    capture_dir = args.output / "capture"
                    capture_dir.mkdir(parents=True, exist_ok=True)
                    (capture_dir / "armed").touch()
                prompts = [
                    "Write a Python function that merges overlapping intervals. Explain its complexity.\n",
                    "Find and fix the bug: def mean(xs): return sum(xs) / (len(xs) - 1)\n",
                    "Explain why a file descriptor leak can make a web server stop accepting connections.\n",
                    "Write a pytest test for a function which removes duplicates while preserving order.\n",
                ]
                if args.history_repeat:
                    history = "Repository history:\n" + "".join(
                        f"Revision {i}: utilities.py defines add(a, b) returning a + b; its tests cover zero and negative values.\n"
                        for i in range(args.history_repeat))
                    prompts = [history + "\nCurrent task:\n" + prompt for prompt in prompts]
                def complete(item):
                    index, prompt = item
                    begin = time.monotonic()
                    result = fetch(url + "/v1/completions", {
                        "model": "qualification", "prompt": prompt, "temperature": 0,
                        "max_tokens": 128, "seed": 104729, "return_token_ids": True},
                        timeout=3600 if args.compression_probe else 180)
                    return {"index": index, "elapsed_seconds": time.monotonic() - begin, "result": result}
                single_prompt = prompts[0]
                if args.compression_probe:
                    history = fetch(url + "/tokenize", {"model": "qualification", "prompt":
                        "Repository history: utilities.py defines add(a, b) returning a + b; tests cover zero and negative values.\n"})["tokens"]
                    task = fetch(url + "/tokenize", {"model": "qualification", "prompt":
                        "\nCurrent task:\n" + prompts[0]})["tokens"]
                    length = 198656  # K+G+2048: crosses the genuine mainline refresh boundary.
                    single_prompt = (history * ((length + len(history) - 1) // len(history)))[:length-len(task)] + task
                    report["long_single_prompt_tokens"] = len(single_prompt)
                report["single"] = complete((0, single_prompt))
                begin = time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    report["concurrent"] = list(executor.map(complete, enumerate(prompts)))
                report["concurrent_wall_seconds"] = time.monotonic() - begin
                report["memory"] = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], text=True).strip()
                if args.capture_probe:
                    capture_report = json.loads((args.output / "capture/capture.json").read_text())
                    if any(r.startswith("_warmup_") for r in capture_report["requests"]):
                        raise RuntimeError("Probe captured warmup instead of a real request")
                    report["capture"] = capture_report
                    if args.scheduled_rebuild_probe and not capture_report.get("scheduled_rebuild", {}).get("all_byte_equal"):
                        raise RuntimeError("Scheduler-driven rebuild did not complete")
                if args.archive_probe:
                    path = args.output / "capture/archive.json"
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        try:
                            archived = json.loads(path.read_text())
                            if len(archived["completed"]) == 5:
                                break
                        except (FileNotFoundError, json.JSONDecodeError):
                            pass
                        time.sleep(0.05)
                    else:
                        raise RuntimeError("Continuous archive did not finalize all five requests")
                    validate_archive(report, archived)
                    report["archive"] = archived
                if args.replay_probe:
                    replay = json.loads((args.output / "capture/replay.json").read_text())
                    if {case["case"] for case in replay["cases"]} != {"initial_prefill", "decode", "four_requests"}:
                        raise RuntimeError("Native replay did not cover all required batch states")
                    if not all(case["output_byte_equal"] and case["state_byte_equal"] for case in replay["cases"]):
                        raise RuntimeError("Native recurrent replay changed model output or state")
                    report["replay"] = replay
                if args.compression_probe:
                    compressed = json.loads((args.output / "capture/compression.json").read_text())
                    if not compressed["refreshes"] or not all(
                            r["output_byte_equal"] and r["state_byte_equal"] and r["offset"] > 0
                            and r["selection_budget"] == 131072 and r["generation_budget"] == 65536
                            for r in compressed["refreshes"]):
                        raise RuntimeError("Real K128K/G64K compressed replay did not pass")
                    report["compression"] = compressed
                report["status"] = "passed"
        except Exception as error:
            report["status"] = "failed"
            report["error"] = repr(error)
            raise
        finally:
            if process is not None:
                try:
                    process.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            (args.output / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
