"""Local GPU prerequisite smoke; deliberately not a KVMem throughput benchmark."""
import argparse
import concurrent.futures
from dataclasses import asdict
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
    if (archived["active_requests"] or archived["archive_used_slots"] or archived["pending_downloads"] or
            archived.get("cpu_cache_used_bytes", 0)):
        raise RuntimeError("Finalized requests retained authority resources")
    block_tokens = archived.get("block_tokens", 128)
    if not all(r["sealed_records_verified"] == r["history_end"] // block_tokens
               and r["history_end"] == r["accepted_tokens"]
               for r in archived["completed"]):
        raise RuntimeError("Incomplete archive verification or accepted-history mismatch")
    if not sum(r["sealed_records_verified"] for r in archived["completed"]):
        raise RuntimeError("Archive qualification did not exercise a complete immutable record")
    items = ([report["single"]] if "single" in report else []) + report["concurrent"]
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


def validate_bounded(report, workers, schedulers, *, require_concurrent=False):
    """Match transactions exactly and prove compression for every long input."""
    def indexed(records):
        result = {(x["request_id"], x["command_id"]): x for x in records}
        if len(result) != len(records):
            raise RuntimeError("Duplicate bounded refresh transaction")
        return result

    by_worker, by_scheduler = indexed(workers), indexed(schedulers)
    if not workers or by_worker.keys() != by_scheduler.keys():
        raise RuntimeError("Bounded worker and scheduler transaction sets differ")
    items = ([report["single"]] if "single" in report else []) + report["concurrent"]
    http_ids = [item["result"]["id"] for item in items]
    for key, worker in by_worker.items():
        scheduler = by_scheduler[key]
        request_id = worker["request_id"]
        if sum(request_id.startswith(value + "-0-") for value in http_ids) != 1:
            raise RuntimeError("Bounded transaction does not belong to the HTTP request set")
        selected, query = worker["selected_tokens"], worker["query_tokens"]
        if (not 0 < selected <= 131072 or query <= 0 or
                selected != scheduler["selected_tokens"] or query != scheduler["replay_tokens"]):
            raise RuntimeError("Worker and scheduler disagree on selected/query extent")
        groups = scheduler["groups"]
        if not groups:
            raise RuntimeError("Bounded transaction has no attention allocation")
        for group in groups:
            if (group["offset"] != worker["history_begin"] - selected or
                    group["blocks"] != (selected + query + group["block_tokens"] - 1) // group["block_tokens"]):
                raise RuntimeError("Native allocation does not match acknowledged working window")
        batch = worker["batch_requests"]
        if request_id not in batch or len(set(batch)) != len(batch):
            raise RuntimeError("Invalid native query batch membership")
    compressed = {w["request_id"] for w in workers if w["history_begin"] > 131072}
    if not compressed:
        raise RuntimeError("Bounded run lacks actual compressed refreshes")
    for item in items:
        response = item["result"]
        # A prompt only slightly above K can finish within its first query;
        # require actual selection when input crosses a full 1024-token chunk.
        if response["usage"]["prompt_tokens"] > 132096 and not any(
                request_id.startswith(response["id"] + "-0-") for request_id in compressed):
            raise RuntimeError("Long HTTP request lacks an actual compressed refresh")
    max_batch = max(len(w["batch_requests"]) for w in workers)
    if require_concurrent:
        concurrent_ids = {item["result"]["id"] for item in report["concurrent"]}
        co_batched = any(all(any(native.startswith(http + "-0-") for native in w["batch_requests"])
                            for http in concurrent_ids) for w in workers)
        if len(concurrent_ids) != 4 or not co_batched:
            raise RuntimeError("Long C4 did not exercise a native four-request refresh batch")
    return {"worker_refreshes": workers, "scheduler_commits": schedulers,
            "compressed_requests": len(compressed), "max_refresh_batch_requests": max_batch}


def read_frozen(path, digest, output_tokens):
    path = Path(path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Frozen input SHA256 does not match its manifest")
    frozen = json.loads(data)
    ids = frozen["input_ids"]
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 262144 - output_tokens or
            any(type(token) is not int or not 0 <= token < 248320 for token in ids)):
        raise ValueError("Frozen Qwen input contains invalid token IDs or exceeds context capacity")
    if (type(frozen["query_begin"]) is not int or type(frozen["query_end"]) is not int or
            not 0 <= frozen["query_begin"] < frozen["query_end"] <= len(ids)):
        raise ValueError("Frozen query boundaries are outside the original input")
    provenance = {"path": str(path.resolve()), "sha256": digest,
        "id": frozen["id"], "tokens": len(ids), "query_begin": frozen["query_begin"],
        "query_end": frozen["query_end"], "tool_execution": False}
    return frozen, provenance


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
    parser.add_argument("--bounded-target", action="store_true",
                        help="Scheduled semantic prefill and bounded target page ownership (no DFlash)")
    parser.add_argument("--cpu-read-cache-mib", type=int, default=0,
                        help="Experimental bounded CPU archive cache; requires bounded-target")
    parser.add_argument("--packed-assembly", action="store_true",
                        help="Use archive-layout GPU assembly; requires bounded-target")
    parser.add_argument("--frozen-input", type=Path,
                        help="Hash-checked native DeepSWE JSON; preserve its exact input IDs")
    parser.add_argument("--frozen-sha256")
    parser.add_argument("--concurrent-frozen-input", nargs=2, action="append", metavar=("PATH", "SHA256"),
                        help="Repeat exactly four times to replace the short C4 fixtures with exact native histories")
    parser.add_argument("--prefill-request-token-cap", type=int, default=0,
                        help="Native vLLM long-prefill token cap per request (0 keeps the default)")
    parser.add_argument("--concurrent-only", action="store_true",
                        help="Run only the four frozen histories; avoid repeating an already qualified C1")
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--stream-metrics", action="store_true",
                        help="Record HTTP token bursts, TTFT and correctly overlapping group rates")
    parser.add_argument("--history-repeat", type=int, default=0,
                        help="Prepend deterministic repository-like history to exercise chunked prefill")
    args = parser.parse_args()
    if not 1 <= args.output_tokens <= 8192:
        parser.error("output-tokens must be in [1, 8192]")
    if args.prefill_request_token_cap < 0 or (args.prefill_request_token_cap and
            (args.prefill_request_token_cap > 1024 or args.prefill_request_token_cap % 128)):
        parser.error("prefill-request-token-cap must be 0 or a multiple of 128 up to the 1024 batch budget")
    if args.cpu_read_cache_mib < 0:
        parser.error("cpu-read-cache-mib must be nonnegative")
    if (args.cpu_read_cache_mib or args.packed_assembly) and not args.bounded_target:
        parser.error("Assembly/cache candidates currently require bounded-target qualification")
    if args.bounded_target:
        if any((args.dflash, args.compression_probe, args.replay_probe, args.rebuild_probe,
                args.scheduled_rebuild_probe, args.history_repeat)):
            parser.error("Bounded target requires its own non-speculative scheduled qualification")
        args.archive_probe = True
    frozen = frozen_provenance = None
    frozen_concurrent = []
    if bool(args.frozen_input) != bool(args.frozen_sha256):
        parser.error("frozen-input and frozen-sha256 must be supplied together")
    if args.concurrent_frozen_input and len(args.concurrent_frozen_input) != 4:
        parser.error("concurrent-frozen-input must specify exactly four path/hash pairs")
    try:
        if args.frozen_input:
            frozen, frozen_provenance = read_frozen(args.frozen_input, args.frozen_sha256, args.output_tokens)
        frozen_concurrent = [read_frozen(path, digest, args.output_tokens)
                             for path, digest in args.concurrent_frozen_input or ()]
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    if args.concurrent_only and (not frozen_concurrent or frozen is not None):
        parser.error("concurrent-only requires four concurrent frozen inputs and no separate single input")
    if frozen is not None or frozen_concurrent:
        if args.history_repeat or args.rebuild_probe or args.scheduled_rebuild_probe or args.replay_probe:
            parser.error("Frozen history is not interchangeable with synthetic/identity probe fixtures")
        if frozen_concurrent and args.compression_probe:
            parser.error("Compression diagnostic has not qualified multi-request refresh")
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
    report = {"scope": scope, "dflash": args.dflash, "concurrent_only": args.concurrent_only,
              "sampling": {"temperature": 0, "seed": 104729, "max_tokens": args.output_tokens,
                           "natural_eos": True}}
    if frozen_provenance is not None:
        report["frozen_input"] = frozen_provenance
    if frozen_concurrent:
        report["frozen_concurrent"] = [provenance for _, provenance in frozen_concurrent]
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
        env = {key: value for key, value in os.environ.items() if not key.startswith("KVMEM_")}
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
        if args.bounded_target:
            env["KVMEM_BOUNDED_TARGET"] = "1"
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
                   "--kv-cache-dtype", "fp8_e4m3", "--max-model-len", "262144" if args.compression_probe or args.bounded_target or frozen is not None or frozen_concurrent else str(max(8192, args.output_tokens + 4096)),
                   "--max-num-seqs", "4", "--max-num-batched-tokens", "1024",
                   "--gpu-memory-utilization", "0.72", "--no-enable-prefix-caching",
                   "--generation-config", "vllm", "--enforce-eager"]
        if args.prefill_request_token_cap:
            command += ["--long-prefill-token-threshold", str(args.prefill_request_token_cap)]
        if args.compression_probe or args.bounded_target or args.scheduled_rebuild_probe or frozen is not None or frozen_concurrent:
            command += ["--no-async-scheduling"]
        if args.compression_probe:
            report["scope"] = "real K128K/G64K target compression/replay; unbounded native allocation; long C1 only"
        if args.bounded_target:
            report["scope"] = "scheduled semantic target refresh and bounded native allocation; " + ("" if args.concurrent_only else "long C1, ") + ("frozen long C4" if frozen_concurrent else "short C4")
        if args.scheduled_rebuild_probe or args.bounded_target:
            command += ["--scheduler-cls", "vllm_kvmem.scheduler.KVMemScheduler",
                        "--kv-transfer-config", json.dumps({
                            "kv_connector": "KVMemConnector", "kv_role": "kv_both",
                            "kv_connector_module_path": "vllm_kvmem.connector",
                            "kv_connector_extra_config": {"kvmem": {
                                "cpu_read_cache_bytes": args.cpu_read_cache_mib * 1024**2,
                                "packed_assembly": args.packed_assembly}}})]
        if args.dflash:
            command += ["--speculative-config", json.dumps({
                "method": "dflash", "model": str(models / "Qwen3.8-27B-DFlash2-BF16"),
                "num_speculative_tokens": 7, "draft_load_config": {"load_format": "safetensors"},
                "kv_cache_dtype": "bfloat16"})]
        report["command"] = command
        report["plugin_options"] = {key: value for key, value in env.items() if key.startswith("KVMEM_")}
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
                def complete(item, phase="concurrent"):
                    index, prompt = item
                    begin = time.monotonic()
                    payload = {
                        "model": "qualification", "prompt": prompt, "temperature": 0,
                        "max_tokens": args.output_tokens, "seed": 104729, "return_token_ids": True}
                    timeout = 3600 if args.compression_probe or args.bounded_target or frozen is not None or frozen_concurrent else 180
                    measured = None
                    if args.stream_metrics:
                        from vllm_kvmem.measurement import stream_completion
                        measured = stream_completion(url + "/v1/completions", payload, timeout=timeout,
                            trace_path=args.output / f"{phase}_{index}_stream.jsonl")
                        result = {"id": measured.response_id, "usage": measured.usage,
                            "choices": [{"index": 0, "text": measured.text, "token_ids": measured.token_ids,
                                         "finish_reason": measured.finish_reason}]}
                    else:
                        result = fetch(url + "/v1/completions", payload, timeout=timeout)
                    entry = {"index": index, "elapsed_seconds": time.monotonic() - begin, "result": result}
                    if measured is not None:
                        entry["trace"] = asdict(measured)
                    return entry
                if not args.concurrent_only:
                    single_prompt = prompts[0]
                    if frozen is not None:
                        single_prompt = frozen["input_ids"]
                        report["long_single_prompt_tokens"] = len(single_prompt)
                    elif args.compression_probe or args.bounded_target:
                        history = fetch(url + "/tokenize", {"model": "qualification", "prompt":
                            "Repository history: utilities.py defines add(a, b) returning a + b; tests cover zero and negative values.\n"})["tokens"]
                        task = fetch(url + "/tokenize", {"model": "qualification", "prompt":
                            "\nCurrent task:\n" + prompts[0]})["tokens"]
                        length = 134144 if args.bounded_target else 198656
                        single_prompt = (history * ((length + len(history) - 1) // len(history)))[:length-len(task)] + task
                        report["long_single_prompt_tokens"] = len(single_prompt)
                    report["single"] = complete((0, single_prompt), phase="single")
                if frozen_concurrent:
                    prompts = [item["input_ids"] for item, _ in frozen_concurrent]
                begin = time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    report["concurrent"] = list(executor.map(complete, enumerate(prompts)))
                report["concurrent_wall_seconds"] = time.monotonic() - begin
                if args.stream_metrics:
                    from vllm_kvmem.measurement import StreamTrace, summarize_group
                    report["http_measurements"] = {
                        "concurrent": summarize_group([StreamTrace(**item["trace"]) for item in report["concurrent"]])}
                    if "single" in report:
                        report["http_measurements"]["single"] = summarize_group([StreamTrace(**report["single"]["trace"])])
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
                    expected_requests = len(report["concurrent"]) + int("single" in report)
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        try:
                            archived = json.loads(path.read_text())
                            if len(archived["completed"]) == expected_requests:
                                break
                        except (FileNotFoundError, json.JSONDecodeError):
                            pass
                        time.sleep(0.05)
                    else:
                        raise RuntimeError("Continuous archive did not finalize every submitted request")
                    validate_archive(report, archived)
                    if args.bounded_target and (archived.get("cpu_cache_capacity_bytes", 0) != args.cpu_read_cache_mib * 1024**2
                            or archived.get("packed_assembly", False) != args.packed_assembly):
                        raise RuntimeError("Worker did not apply requested cache/assembly options")
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
                if args.bounded_target:
                    workers = [json.loads(line) for line in (args.output / "capture/bounded_refresh.jsonl").read_text().splitlines()]
                    schedulers = [json.loads(line) for line in (args.output / "capture/scheduler_refresh.jsonl").read_text().splitlines()]
                    report["bounded"] = validate_bounded(report, workers, schedulers,
                        require_concurrent=bool(args.concurrent_frozen_input))
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
