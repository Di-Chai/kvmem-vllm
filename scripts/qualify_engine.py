"""Local GPU prerequisite smoke; deliberately not a KVMem throughput benchmark."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request


def fetch(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dflash", action="store_true")
    parser.add_argument("--capture-probe", action="store_true")
    parser.add_argument("--rebuild-probe", action="store_true")
    parser.add_argument("--scheduled-rebuild-probe", action="store_true")
    args = parser.parse_args()
    if args.rebuild_probe or args.scheduled_rebuild_probe:
        args.capture_probe = True
    if args.scheduled_rebuild_probe and (args.dflash or args.rebuild_probe):
        parser.error("Scheduled identity probe currently requires target-only execution")
    args.output.mkdir(parents=True, exist_ok=False)
    models = args.models.resolve()
    scope = ("scheduler/worker identity-refresh diagnostic; no compressed KVMem serving"
             if args.scheduled_rebuild_probe else
             "native identity-refresh diagnostic; no compressed KVMem serving"
             if args.rebuild_probe else "engine prerequisite, no KVMem enabled")
    report = {"scope": scope, "dflash": args.dflash}
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
                   "--kv-cache-dtype", "fp8_e4m3", "--max-model-len", "8192",
                   "--max-num-seqs", "4", "--max-num-batched-tokens", "1024",
                   "--gpu-memory-utilization", "0.72", "--no-enable-prefix-caching",
                   "--generation-config", "vllm", "--enforce-eager"]
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
                def complete(item):
                    index, prompt = item
                    begin = time.monotonic()
                    result = fetch(url + "/v1/completions", {
                        "model": "qualification", "prompt": prompt, "temperature": 0,
                        "max_tokens": 128, "seed": 104729, "return_token_ids": True})
                    return {"index": index, "elapsed_seconds": time.monotonic() - begin, "result": result}
                report["single"] = complete((0, prompts[0]))
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
