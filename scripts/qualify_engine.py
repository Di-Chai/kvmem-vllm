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
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import uuid


def freeze_worker_package(root, output, manifest):
    """Snapshot late-imported worker code and the built extension before launch."""
    source = root / 'src' / 'vllm_kvmem'
    target = output / 'worker_source' / 'vllm_kvmem'
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    hashes = {}
    for path in sorted(target.rglob('*')):
        if not path.is_file():
            continue
        name = path.relative_to(target).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        tracked = 'src/vllm_kvmem/' + name
        expected = manifest.get(tracked)
        if ((expected is not None and digest != expected) or
                hashlib.sha256((source / name).read_bytes()).hexdigest() != digest or
                (path.suffix == '.py' and expected is None)):
            raise RuntimeError(f'Worker source changed while freezing: {name}')
        hashes[name] = digest
    expected_python = {name.removeprefix('src/vllm_kvmem/') for name in manifest
                       if name.startswith('src/vllm_kvmem/') and name.endswith('.py')}
    if expected_python != {name for name in hashes if name.endswith('.py')}:
        raise RuntimeError('Worker source file set changed while freezing')
    (output / 'worker_manifest.json').write_text(json.dumps(hashes, indent=2))
    return target.parent.resolve(), hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def fetch(url, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def stop_owned_server(process):
    """Stop our start_new_session child before joining blocked HTTP readers."""
    if process is None or process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def validate_archive(report, archived):
    if (archived["active_requests"] or archived.get('retiring_requests', 0) or
            archived["archive_used_slots"] or archived["pending_downloads"] or
            archived.get("cpu_cache_used_bytes", 0)):
        raise RuntimeError("Finalized requests retained authority resources")
    block_tokens = archived.get("block_tokens", 128)
    mode = archived.get("archive_verification", "full_checksum")
    if mode not in ("full_checksum", "publication_only"):
        raise RuntimeError("Unknown archive verification mode")
    if not all(r.get("sealed_records_archived", r["sealed_records_verified"]) == r["history_end"] // block_tokens
               and r["history_end"] == r["accepted_tokens"]
               and r["sealed_records_verified"] == (r["history_end"] // block_tokens if mode == "full_checksum" else 0)
               for r in archived["completed"]):
        raise RuntimeError("Incomplete archive verification or accepted-history mismatch")
    if not sum(r.get("sealed_records_archived", r["sealed_records_verified"]) for r in archived["completed"]):
        raise RuntimeError("Archive qualification did not exercise a complete immutable record")
    items = ([report["single"]] if "single" in report else []) + report["concurrent"]
    if len(items) + len(report.get("cancelled", [])) != len(archived["completed"]):
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
    for cancelled in report.get("cancelled", []):
        matched = [record for record in archived["completed"]
                   if record["request_id"].startswith(cancelled["request_id"] + "-0-")]
        if len(matched) != 1:
            raise RuntimeError("Missing or ambiguous cancelled request authority")
        record = matched[0]
        if not cancelled["trigger"]["history_begin"] <= record["history_end"] < cancelled["prompt_tokens"]:
            raise RuntimeError("Cancelled prefill did not stop with its captured accepted prefix")
        step = record["last_steps"][-1]
        if step["history_begin"] + step["accepted"] != record["history_end"]:
            raise RuntimeError("Cancelled authority disagrees with its last native packet")
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
    items += [{"result": {"id": item["request_id"], "usage": {"prompt_tokens": item["prompt_tokens"]}}}
              for item in report.get("cancelled", [])]
    for key, worker in by_worker.items():
        scheduler = by_scheduler[key]
        request_id = worker["request_id"]
        matched = [item["result"] for item in items if request_id.startswith(item["result"]["id"] + "-0-")]
        if len(matched) != 1:
            raise RuntimeError("Bounded transaction does not belong to the HTTP request set")
        selected, query = worker["selected_tokens"], worker["query_tokens"]
        if (not 0 < selected <= 131072 or query <= 0 or
                selected != scheduler["selected_tokens"] or query != scheduler["replay_tokens"]):
            raise RuntimeError("Worker and scheduler disagree on selected/query extent")
        if report.get("dflash"):
            prompt = matched[0]["usage"]["prompt_tokens"]
            expected_probe = min(query, max(1, prompt - worker["history_begin"]))
            if (not worker.get("draft_groups") or worker.get("query_score_tokens") != expected_probe or
                    (worker["history_begin"] >= prompt and query > 8)):
                raise RuntimeError("Bounded DFlash reselection used unverified draft rows or lost draft groups")
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
        co_batched = any(w["history_begin"] > 131072 and all(any(native.startswith(http + "-0-") for native in w["batch_requests"])
                            for http in concurrent_ids) for w in workers)
        if len(concurrent_ids) != 4 or not co_batched:
            raise RuntimeError("Long C4 did not exercise a native four-request refresh batch")
    return {"worker_refreshes": workers, "scheduler_commits": schedulers,
            "compressed_requests": len(compressed), "max_refresh_batch_requests": max_batch}


def validate_cancellation(cancelled, workers, finished):
    for item in cancelled:
        trigger = item["trigger"]
        matches = [worker for worker in workers if worker["request_id"] == trigger["request_id"]
                   and worker["command_id"] == trigger["command_id"]]
        if len(matches) != 1:
            raise RuntimeError("Cancellation trigger lacks a completed native refresh")
        refresh = matches[0]
        if not refresh["refresh_started_at"] <= item["cancelled_at"] <= refresh["refresh_finished_at"]:
            raise RuntimeError("HTTP disconnect did not occur during the native refresh transaction")
        statuses = [entry for entry in finished if entry["request_id"] == trigger["request_id"]]
        if len(statuses) != 1 or statuses[0]["status"] != "FINISHED_ABORTED":
            raise RuntimeError("Native scheduler did not acknowledge client cancellation")
        if statuses[0]["at"] < item["cancelled_at"]:
            raise RuntimeError("Native request ended before client cancellation")


def validate_released_scheduler(snapshot):
    if (snapshot["requests"] or snapshot["refresh_pending"] or snapshot["connector_pending"] or
            snapshot["reservations"] or snapshot["owned_blocks"] or
            snapshot["free_blocks"] != snapshot["total_blocks"] - 1):
        raise RuntimeError("Native scheduler retained requests, reservations or GPU pages")


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
    parser.add_argument("--bounded-dflash", action="store_true",
                        help="Experimental synchronous DFlash2 with scheduled bounded semantic refresh")
    parser.add_argument("--bounded-target", action="store_true",
                        help="Scheduled semantic prefill and bounded target page ownership (no DFlash)")
    parser.add_argument("--cpu-read-cache-mib", type=int, default=0,
                        help="Experimental bounded CPU archive cache; requires bounded-target")
    parser.add_argument("--packed-assembly", action="store_true",
                        help="Use archive-layout GPU assembly; requires bounded-target")
    parser.add_argument("--direct-pinned-reads", action="store_true",
                        help="Experimental archive reads directly into bounded pinned H2D slots; requires packed-assembly")
    parser.add_argument("--frozen-input", type=Path,
                        help="Hash-checked native DeepSWE JSON; preserve its exact input IDs")
    parser.add_argument("--frozen-sha256")
    parser.add_argument("--concurrent-frozen-input", nargs=2, action="append", metavar=("PATH", "SHA256"),
                        help="Repeat exactly four times to replace the short C4 fixtures with exact native histories")
    parser.add_argument("--batch-token-budget", type=int, default=1024,
                        help="Native token budget per GPU batch; does not change K/G")
    parser.add_argument("--prefill-request-token-cap", type=int, default=0,
                        help="Native vLLM long-prefill token cap per request (0 keeps the default)")
    parser.add_argument("--prefill-cadence", type=int, default=1,
                        help="Experimental native prefill release cadence during mixed decode, default every step")
    parser.add_argument("--kv-cache-memory-mib", type=int, default=0,
                        help="Explicit native GPU KV pool budget; 0 uses native utilization-based sizing")
    parser.add_argument("--require-capacity-pressure", action="store_true",
                        help="Long C4 admission test: require rejected reservations and complete physical cleanup")
    parser.add_argument("--cancel-during-refresh", action="store_true",
                        help="Disconnect the frozen C1 during a real compressed refresh, then run the four survivors")
    parser.add_argument("--concurrent-only", action="store_true",
                        help="Run only the four frozen histories; avoid repeating an already qualified C1")
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--skip-archive-readback", action="store_true",
                        help="Measure serving without diagnostic finish checksums; still drain and check publication/cleanup")
    parser.add_argument("--async-archive-retirement", action="store_true",
                        help="Experimental CPU retirement with native completion acknowledgments and bounded authority admission")
    parser.add_argument("--stream-metrics", action="store_true",
                        help="Record HTTP token bursts, TTFT and correctly overlapping group rates")
    parser.add_argument("--history-repeat", type=int, default=0,
                        help="Prepend deterministic repository-like history to exercise chunked prefill")
    args = parser.parse_args()
    if args.bounded_dflash:
        args.bounded_target = args.dflash = True
    if not 1 <= args.output_tokens <= 8192:
        parser.error("output-tokens must be in [1, 8192]")
    if not 128 <= args.batch_token_budget <= 65536 or args.batch_token_budget % 128:
        parser.error("batch-token-budget must be a multiple of 128 within [128, 65536]")
    if args.prefill_request_token_cap < 0 or (args.prefill_request_token_cap and
            (args.prefill_request_token_cap > args.batch_token_budget or args.prefill_request_token_cap % 128)):
        parser.error("prefill-request-token-cap must be 0 or a multiple of 128 up to the native batch budget")
    if args.cpu_read_cache_mib < 0:
        parser.error("cpu-read-cache-mib must be nonnegative")
    if args.direct_pinned_reads and not args.packed_assembly:
        parser.error("direct-pinned-reads requires packed-assembly")
    if not 1 <= args.prefill_cadence <= 64 or (args.prefill_cadence != 1 and not args.bounded_target):
        parser.error("prefill-cadence must be in [1, 64], and nondefault values require bounded serving")
    if (args.cpu_read_cache_mib or args.packed_assembly) and not args.bounded_target:
        parser.error("Assembly/cache candidates currently require bounded-target qualification")
    if args.bounded_target:
        if any((args.dflash and not args.bounded_dflash, args.compression_probe, args.replay_probe, args.rebuild_probe,
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
    if args.kv_cache_memory_mib < 0:
        parser.error("kv-cache-memory-mib must be nonnegative")
    if args.require_capacity_pressure and (not args.bounded_target or not args.concurrent_only or
            not frozen_concurrent or not args.kv_cache_memory_mib):
        parser.error("capacity pressure requires bounded concurrent-only frozen C4 and an explicit native KV pool budget")
    if args.cancel_during_refresh and (not args.bounded_target or frozen is None or args.concurrent_only):
        parser.error("cancel-during-refresh requires bounded serving, a frozen C1, and the subsequent C4 recovery group")
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
    if args.skip_archive_readback and not args.bounded_target:
        parser.error("skip-archive-readback requires bounded serving and its public connector options")
    if args.async_archive_retirement and not args.bounded_target:
        parser.error("async-archive-retirement requires bounded serving")
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
    worker_source, worker_hash = freeze_worker_package(root, args.output, manifest)
    sys.path.insert(0, str(worker_source))  # Freeze late-imported client measurement helpers too.
    report["worker_source"] = str(worker_source)
    report["worker_manifest_sha256"] = worker_hash
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
        env["PYTHONPATH"] = str(worker_source) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        imported = subprocess.check_output([sys.executable, "-c",
            "import vllm_kvmem; print(vllm_kvmem.__file__)"], env=env, text=True).strip()
        if Path(imported).resolve() != worker_source / 'vllm_kvmem' / '__init__.py':
            raise RuntimeError('Worker package snapshot did not take precedence over installed code')
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
        if args.cancel_during_refresh or args.require_capacity_pressure or args.async_archive_retirement:
            env["KVMEM_LIFECYCLE_TRACE"] = "1"
        if args.bounded_dflash:
            env["KVMEM_BOUNDED_DFLASH"] = "1"
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
                   "--max-num-seqs", "4", "--max-num-batched-tokens", str(args.batch_token_budget),
                   "--gpu-memory-utilization", "0.72", "--no-enable-prefix-caching",
                   "--generation-config", "vllm", "--enforce-eager"]
        if args.kv_cache_memory_mib:
            command += ["--kv-cache-memory-bytes", str(args.kv_cache_memory_mib * 1024**2)]
        if args.prefill_request_token_cap:
            command += ["--long-prefill-token-threshold", str(args.prefill_request_token_cap)]
        if args.compression_probe or args.bounded_target or args.scheduled_rebuild_probe or frozen is not None or frozen_concurrent:
            command += ["--no-async-scheduling"]
        if args.compression_probe:
            report["scope"] = "real K128K/G64K target compression/replay; unbounded native allocation; long C1 only"
        if args.bounded_target:
            report["scope"] = ("experimental DFlash2; " if args.bounded_dflash else "target-only; ") + "scheduled semantic refresh and bounded native allocation; " + ("" if args.concurrent_only else "long C1, ") + ("frozen long C4" if frozen_concurrent else "short C4")
        if args.scheduled_rebuild_probe or args.bounded_target:
            command += ["--scheduler-cls", "vllm_kvmem.scheduler.KVMemScheduler",
                        "--kv-transfer-config", json.dumps({
                            "kv_connector": "KVMemConnector", "kv_role": "kv_both",
                            "kv_connector_module_path": "vllm_kvmem.connector",
                            "kv_connector_extra_config": {"kvmem": {
                                "cpu_read_cache_bytes": args.cpu_read_cache_mib * 1024**2,
                                "packed_assembly": args.packed_assembly,
                                "direct_pinned_reads": args.direct_pinned_reads,
                                "prefill_cadence": args.prefill_cadence,
                                "async_archive_retirement": args.async_archive_retirement,
                                "verify_archive_on_finish": not args.skip_archive_readback}}})]
        if args.dflash:
            command += ["--speculative-config", json.dumps({
                "method": "dflash", "model": str(models / "Qwen3.8-27B-DFlash2-BF16"),
                "num_speculative_tokens": 7, "draft_load_config": {"load_format": "safetensors"},
                "kv_cache_dtype": "bfloat16"})]
        report["command"] = command
        report["plugin_options"] = {key: value for key, value in env.items() if key.startswith("KVMEM_")}
        (args.output / "command.json").write_text(json.dumps(report, indent=2))
        process = None
        monitor = None
        try:
            with (args.output / "server.log").open("w") as logfile:
                process = subprocess.Popen(command, env=env, stdout=logfile, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                report["server_pid"] = process.pid
                from vllm_kvmem.telemetry import ResourceMonitor
                monitor = ResourceMonitor(args.output / "resources.jsonl", process.pid).start()
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
                    if args.cancel_during_refresh:
                        from vllm_kvmem.cancellation import cancel_during_refresh
                        payload = {"model": "qualification", "prompt": single_prompt,
                            "request_id": "kvmem-cancel-" + uuid.uuid4().hex,
                            "temperature": 0, "seed": 104729, "max_tokens": args.output_tokens}
                        report["cancelled"] = [cancel_during_refresh(url + "/v1/completions", payload,
                            args.output / "capture/bounded_active.json", args.output / "cancelled_stream.jsonl")]
                    else:
                        report["single"] = complete((0, single_prompt), phase="single")
                if frozen_concurrent:
                    prompts = [item["input_ids"] for item, _ in frozen_concurrent]
                begin = time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    try:
                        report["concurrent"] = list(executor.map(complete, enumerate(prompts)))
                    except BaseException:
                        # Executor.__exit__ otherwise waits for still-streaming
                        # clients before the outer finally can stop the engine.
                        stop_owned_server(process)
                        raise
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
                    expected_requests = len(report["concurrent"]) + int("single" in report) + len(report.get("cancelled", []))
                    retirement_start = time.monotonic()
                    deadline = retirement_start + 180
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
                    report['archive_drain_seconds_after_http'] = time.monotonic() - retirement_start
                    if args.bounded_target and (archived.get("cpu_cache_capacity_bytes", 0) != args.cpu_read_cache_mib * 1024**2
                            or archived.get("packed_assembly", False) != args.packed_assembly
                            or archived.get("direct_pinned_reads", False) != args.direct_pinned_reads
                            or archived.get("async_archive_retirement", False) != args.async_archive_retirement):
                        raise RuntimeError("Worker did not apply requested cache/assembly options")
                    expected_verification = "publication_only" if args.skip_archive_readback else "full_checksum"
                    if archived.get("archive_verification", "full_checksum") != expected_verification:
                        raise RuntimeError("Worker did not apply requested archive verification mode")
                    report["archive_checksum_qualified"] = expected_verification == "full_checksum"
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
                    scheduled_queries = json.loads((args.output / 'capture/scheduled_queries.json').read_text())
                    if scheduled_queries['prefill_cadence'] != args.prefill_cadence:
                        raise RuntimeError('Scheduler did not apply the requested prefill cadence')
                    if args.prefill_request_token_cap and any(int(size) > args.prefill_request_token_cap
                            for size in scheduled_queries['queries']['prefill']):
                        raise RuntimeError('Actual scheduled prefill exceeds the requested per-request query cap')
                    report['scheduled_queries'] = scheduled_queries
                    workers = [json.loads(line) for line in (args.output / "capture/bounded_refresh.jsonl").read_text().splitlines()]
                    schedulers = [json.loads(line) for line in (args.output / "capture/scheduler_refresh.jsonl").read_text().splitlines()]
                    if args.cancel_during_refresh:
                        finished = [json.loads(line) for line in (args.output / "capture/scheduler_finished.jsonl").read_text().splitlines()]
                        validate_cancellation(report["cancelled"], workers, finished)
                        report["scheduler_finished"] = finished
                    if args.cancel_during_refresh or args.require_capacity_pressure or args.async_archive_retirement:
                        # The worker publishes retirement before its completion
                        # reaches the scheduler. Wait for native physical release.
                        deadline = time.monotonic() + 10
                        while True:
                            try:
                                released = json.loads((args.output / "capture/scheduler_resources.json").read_text())
                                validate_released_scheduler(released)
                                break
                            except (RuntimeError, FileNotFoundError, json.JSONDecodeError):
                                if time.monotonic() >= deadline:
                                    raise
                                time.sleep(0.01)
                        report["scheduler_resources"] = released
                        if args.require_capacity_pressure and not released["reservation_rejections"]:
                            raise RuntimeError("Requested capacity test never exercised admission backpressure")
                    report["bounded"] = validate_bounded(report, workers, schedulers,
                        require_concurrent=bool(args.concurrent_frozen_input) and not args.require_capacity_pressure)
                report["status"] = "passed"
        except BaseException as error:
            report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            report["error"] = repr(error)
            raise
        finally:
            if monitor is not None:
                report["resources"] = monitor.stop()
            stop_owned_server(process)
            (args.output / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
