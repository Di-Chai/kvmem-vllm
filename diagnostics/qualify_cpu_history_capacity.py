"""Serial full-history capacity probe against an already running cloud server.

Synthetic token history tests storage/admission/restore, not semantic quality.
Every extension preserves the exact previously accepted token IDs. The final
emitted token is not normally forwarded: logical 262144 means durable KV262143.
"""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import httpx


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read(path):
    return json.loads(path.read_text())


def quiescent(state):
    history = state.get("cpu_history_admission", {})
    owners = state.get("session_owners")
    return (
        all(state.get(k) == 0 for k in
            ("requests", "refresh_pending", "connector_pending", "owned_blocks"))
        and state.get("reservations") == []
        and state.get("publication_blocked") == []
        and all(history.get(k) == 0 for k in ("active", "retiring", "waiting"))
        and isinstance(owners, list)
        and all(owner.get("phase") == "idle" for owner in owners)
    )


def idle(capture, timeout=120, *, after_at=None, empty=False, request_id=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            state = read(capture / "scheduler_resources.json")
            if (quiescent(state) and (after_at is None or state.get("at", -1) >= after_at)
                    and (not empty or (
                        state["session_owners"] == [] and state.get("session_views") == []
                        and state.get("sessions", {}).get("entries") == 0
                        and state["cpu_history_admission"].get("reserved_records") == 0
                        and state.get("free_blocks") == state.get("total_blocks", 0) - 1))):
                if request_id is not None:
                    completed = completion_for(read(capture / "archive.json"), request_id)
                    if completed.get("retirement_ack_ready_at", -1) < (after_at or 0):
                        raise AssertionError("Terminal archive snapshot predates this stage")
                return state
        except (OSError, ValueError, AssertionError):
            pass
        time.sleep(0.2)
    raise RuntimeError("Scheduler did not settle into the required idle ownership state")


def initial_state(capture, client, origin):
    """A fresh engine publishes its first scheduler snapshot only after work."""
    client.get(origin + "/health", timeout=30).raise_for_status()
    if (capture / "scheduler_resources.json").exists():
        state = idle(capture, empty=True)
        archive = read(capture / "archive.json")
        return state, archive, "observed_empty_scheduler"
    # This is only the fresh, exclusively owned server path. Historical work
    # without its resource snapshot must never be presented as an empty ledger.
    archive = read(capture / "archive.json") if (capture / "archive.json").exists() else {}
    if any(archive.get(key, 0) for key in (
            "completed_requests", "active_requests", "retiring_requests",
            "session_checkpoints", "session_retiring_requests", "archive_used_slots")):
        raise AssertionError("Missing scheduler snapshot after prior archive activity")
    for name in ("scheduler_finished.jsonl", "async_refresh.jsonl", "async_restore.jsonl"):
        path = capture / name
        if path.exists() and path.stat().st_size:
            raise AssertionError("Missing scheduler snapshot after prior request activity")
    return None, archive, "healthy_fresh_engine_snapshot_not_yet_published"


def fingerprint(tokens):
    packed = array("I", tokens)
    if packed.itemsize != 4:
        raise ValueError("Token fingerprint requires native uint32 storage")
    return hashlib.sha256(packed.tobytes()).hexdigest()


def completion_for(archive, request_id):
    if not isinstance(request_id, str) or not request_id:
        raise AssertionError("Stream supplied no request identity")
    rows = [row for row in archive.get("completed", [])
            if row.get("request_id") == request_id
            or row.get("request_id", "").startswith(request_id + "-")]
    if len(rows) != 1:
        raise AssertionError("Archive lacks this request's unique terminal completion")
    return rows[0]


def restore_for(capture, request_id):
    path = capture / "async_restore.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    matches = [row for row in rows if row.get("request_id") == request_id]
    if len(matches) != 1 or matches[0].get("cancelled") is not False:
        raise AssertionError("Continuation lacks one successful native restore event")
    return matches[0]


def validate_stage(state, archive, stage, history, session, *, block_tokens,
                   record_bytes, expected_hits, expected_restored_tokens,
                   previous_retained, initial_retired, capture):
    if not quiescent(state):
        raise AssertionError("Stage still owns running or retiring resources")
    namespace = [session, session]
    owners = state["session_owners"]
    if len(owners) != 1 or owners[0].get("namespace") != namespace:
        raise AssertionError("Capacity evidence is not exclusive to the probe owner")
    views = state.get("session_views", [])
    if len(views) != 1 or views[0].get("namespace") != namespace:
        raise AssertionError("Expected the probe's retained CPU session checkpoint")
    view = views[0]
    expected = len(history) - 1
    terminal = completion_for(archive, stage["request_id"])
    prefix = terminal.get("terminal_prefix", {})
    if (view.get("history") != expected or not 0 < view.get("window", 0) <= 53248
            or view.get("prefix_sha256") != fingerprint(history[:expected])
            or terminal.get("history_end") != expected
            or terminal.get("checkpoint_tokens") != expected
            or terminal.get("checkpoint_kind") != "M"
            or prefix.get("history_end") != expected
            or prefix.get("output_tokens") != len(stage["output_ids"])
            or prefix.get("status") != "FINISHED_LENGTH_CAPPED"):
        raise AssertionError("Durable terminal KV does not match the exact accepted prefix")
    required_records = (expected + block_tokens - 1) // block_tokens
    minimum_archive_bytes = expected // block_tokens * record_bytes
    if (view.get("allocated_records", -1) < required_records
            or archive.get("archive_backend") != "cpu"
            or archive.get("cpu_native_payloads") is not True
            or archive.get("cpu_authority_bytes", -1) < minimum_archive_bytes
            or archive.get("historical_gc", {}).get("retired") != initial_retired):
        raise AssertionError("CPU storage does not prove full retained history without GC deletion")
    if (state.get("sessions", {}).get("hits") != expected_hits
            or archive.get("session_restored_tokens") != expected_restored_tokens):
        raise AssertionError("Continuation did not reuse the expected previous CPU checkpoint")
    restore = None
    if previous_retained:
        restore = restore_for(capture, terminal["request_id"])
        if restore.get("history_tokens") != previous_retained:
            raise AssertionError("Restore did not consume the complete previous retained prefix")
    return dict(
        accepted_history_tokens=len(history), retained_history_tokens=expected,
        unforwarded_final_output_tokens=1, retained_prefix_sha256=view["prefix_sha256"],
        retained_window_tokens=view["window"], retained_allocated_records=view["allocated_records"],
        minimum_history_records=required_records, minimum_archive_bytes=minimum_archive_bytes,
        cpu_authority_bytes=archive["cpu_authority_bytes"],
        native_request_id=terminal["request_id"], terminal_completion=terminal,
        restore_event=restore, session_hits=expected_hits,
        session_restored_tokens=expected_restored_tokens,
        historical_gc_retired_delta=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-run", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", type=int, nargs="+", default=[65536, 245760, 262112])
    parser.add_argument("--output-tokens", type=int, default=32)
    args = parser.parse_args()
    profile = read(args.server_run / "launch.json")["profile"]
    options = profile["kvmem"]
    if (profile["max_num_seqs"] != 1 or profile["max_model_len"] != 262144
            or options["select_budget"] != 36864 or options["gen_budget"] != 16384
            or options.get("archive_backend") != "cpu"
            or options.get("session_capacity") != 1
            or options.get("session_checkpoint_policy") != "pm"
            or options.get("session_capture_terminal") is not True
            or options.get("cpu_history_coordination") is not True
            or options.get("historical_gc", {}).get("policy") != "statistical"):
        raise ValueError("This probe requires C1/K36K/G16K/logical256K with CPU history, P/M and statistical GC")
    if (not args.targets or args.targets != sorted(set(args.targets))
            or any(n <= 0 or n + args.output_tokens > profile["max_model_len"] for n in args.targets)
            or not 1 <= args.output_tokens <= options["gen_budget"]):
        raise ValueError("Targets must be increasing positive prompt lengths with output headroom")
    base = read(args.input)["input_ids"]
    if not base or any(type(t) is not int or not 0 <= t < 2**32 for t in base):
        raise ValueError("Expected a nonempty frozen uint32 token list")
    capture = args.server_run / "capture"
    geometry = read(capture / "cache-geometry.json")
    row_bytes = geometry.get("cpu_archive_bytes_per_token")
    if (type(row_bytes) is not int or row_bytes <= 0
            or geometry.get("select_budget") != 36864 or geometry.get("gen_budget") != 16384):
        raise ValueError("Missing or mismatched actual CPU archive geometry")
    args.output.mkdir(parents=True, exist_ok=False)
    session = "capacity-256k-" + uuid.uuid4().hex
    origin = f"http://{profile['host']}:{profile['port']}"
    url = origin + "/v1/completions"
    report = dict(scope="Synthetic CPU-history capacity and exact-prefix continuation, not quality or GC throughput",
                  session=session, targets=args.targets, output_tokens=args.output_tokens,
                  profile=profile, geometry=geometry,
                  input_source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  started_unix=time.time(), stages=[], status="running")
    save(args.output / "manifest.json", report)
    history = []
    error = None
    submitted = False
    with httpx.Client(timeout=900, trust_env=False) as client:
        try:
            initial, baseline, mode = initial_state(capture, client, origin)
            save(args.output / "initial-state.json", dict(mode=mode, resources=initial, archive=baseline))
            initial_hits = initial["sessions"]["hits"] if initial is not None else 0
            restored_tokens = baseline.get("session_restored_tokens", 0)
            initial_retired = baseline.get("historical_gc", {}).get("retired", 0)
            previous_retained = 0
            for index, target in enumerate(args.targets, 1):
                need = target - len(history)
                if need < 0:
                    raise ValueError("A target would discard already accepted output")
                prompt = history + (base * ((need + len(base) - 1) // len(base)))[:need]
                payload = dict(model=profile["served_model_name"], prompt=prompt,
                               session_id=session, cache_salt=session,
                               max_tokens=args.output_tokens, temperature=0, seed=19,
                               ignore_eos=True, return_token_ids=True, stream=True,
                               stream_options={"include_usage": True},
                               kv_transfer_params={"kvmem_request_plan": {
                                   "kvmem_reselect": "force", "kvmem_mandatory_parts": "none",
                                   "kvmem_query_span": [max(0, len(prompt)-20), len(prompt)]}})
                stem = f"{index:02d}-{target}"
                save(args.output / (stem + "-request.json"), payload)
                stage = dict(prompt_tokens=target, previous_accepted_prefix_tokens=len(history),
                             exact_previous_prefix_preserved=prompt[:len(history)] == history,
                             started_unix=time.time(), output_ids=[])
                start = time.perf_counter()
                submitted = True
                with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    with (args.output / (stem + "-sse.jsonl")).open("x") as stream:
                        for line in response.iter_lines():
                            if not line.startswith("data: ") or line == "data: [DONE]":
                                continue
                            event = json.loads(line[6:])
                            stream.write(json.dumps(event) + "\n")
                            if event.get("error"):
                                raise RuntimeError(str(event["error"]))
                            stage["request_id"] = event.get("id", stage.get("request_id"))
                            if event.get("usage"):
                                stage["usage"] = event["usage"]
                            for choice in event.get("choices", []):
                                stage["output_ids"].extend(choice.get("token_ids") or [])
                stage["elapsed_seconds"] = time.perf_counter() - start
                if (len(stage["output_ids"]) != args.output_tokens
                        or any(type(t) is not int or not 0 <= t < 2**32 for t in stage["output_ids"])):
                    raise AssertionError("Accepted output IDs differ from the declared probe bound")
                history = prompt + stage["output_ids"]
                state = idle(capture, after_at=start, request_id=stage.get("request_id"))
                archive = read(capture / "archive.json")
                save(args.output / (stem + "-resources.json"), state)
                save(args.output / (stem + "-archive.json"), archive)
                restored_tokens += previous_retained
                stage.update(validate_stage(
                    state, archive, stage, history, session,
                    block_tokens=options["block_tokens"], record_bytes=row_bytes * options["block_tokens"],
                    expected_hits=initial_hits + index - 1, expected_restored_tokens=restored_tokens,
                    previous_retained=previous_retained, initial_retired=initial_retired, capture=capture))
                previous_retained = stage["retained_history_tokens"]
                save(args.output / (stem + "-result.json"), stage)
                report["stages"].append(stage)
                save(args.output / "manifest.json", report)
                print(json.dumps({k:v for k,v in stage.items()
                                  if k not in ("output_ids", "terminal_completion", "restore_event")}), flush=True)
            report["status"] = "passed"
        except Exception as exc:
            error = exc
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            if submitted:
                try:
                    try:
                        state = idle(capture)
                        save(args.output / "before-release-resources.json", state)
                    except Exception as exc:
                        # Still send the owned namespace's release after a failed
                        # probe; the native protocol waits for its fenced cleanup.
                        report["before_release_snapshot_error"] = f"{type(exc).__name__}: {exc}"
                    release = dict(model=profile["served_model_name"], prompt=[1],
                                   session_id=session, cache_salt=session, max_tokens=1,
                                   temperature=0, kvmem_reselect="off",
                                   kvmem_mandatory_parts="none", kvmem_mandatory_spans=[],
                                   kv_transfer_params={"kvmem_release_session": True})
                    release_start = time.perf_counter()
                    response = client.post(url, json=release, timeout=180)
                    save(args.output / "release.json", dict(status=response.status_code, body=response.json()))
                    response.raise_for_status()
                    after = idle(capture, after_at=release_start, empty=True)
                    save(args.output / "after-release-resources.json", after)
                    save(args.output / "after-release-archive.json", read(capture / "archive.json"))
                    report["release_owner_absent"] = True
                except Exception as exc:
                    report["release_error"] = f"{type(exc).__name__}: {exc}"
                    report["status"] = "failed"
                    error = error or exc
            report["finished_unix"] = time.time()
            save(args.output / "manifest.json", report)
    return 1 if error else 0


if __name__ == "__main__":
    sys.exit(main())
