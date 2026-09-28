"""HTTP token-arrival measurements; never sum overlapping request durations.

Speculative tokens may arrive in a single burst. Arrival rates and burst gaps are
service observations, not native kernel timing or individual-token GPU latency.
"""
from dataclasses import dataclass, field
import json
import math
import time
import urllib.request


@dataclass
class StreamTrace:
    submitted: float
    finished: float | None = None
    bursts: list[tuple[float, list[int]]] = field(default_factory=list)
    text: str = ""
    finish_reason: str | None = None
    usage: dict | None = None
    done: bool = False
    response_id: str | None = None

    @property
    def token_ids(self):
        return [token for _, ids in self.bursts for token in ids]

    def validate(self):
        if (not self.done or self.finished is None or self.usage is None or self.finish_reason is None):
            raise ValueError("Incomplete HTTP stream cannot qualify a throughput result")
        times = [self.submitted, *(at for at, _ in self.bursts), self.finished]
        if any(not math.isfinite(at) for at in times) or times != sorted(times):
            raise ValueError("Nonmonotonic token-arrival timestamps")
        if len(self.token_ids) != self.usage["completion_tokens"]:
            raise ValueError("Stream token IDs disagree with final API token accounting")


def stream_completion(url, payload, *, timeout=3600, trace_path=None):
    """One request, no hidden retries; preserve raw events with monotonic times."""
    if payload.get("n", 1) != 1 or payload.get("echo", False):
        raise ValueError("Measurements require one un-echoed completion per request")
    body = {**payload, "stream": True, "return_token_ids": True,
            "stream_options": {"include_usage": True}}
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                    headers={"Content-Type": "application/json"})
    trace = StreamTrace(time.perf_counter())
    output = trace_path.open("x") if trace_path is not None else None
    try:
        if output:
            output.write(json.dumps({"event": "submitted", "at": trace.submitted}) + "\n")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = []
            for line in response:
                line = line.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    data.append(line[5:].lstrip())
                    continue
                if line or not data:
                    continue
                raw, data = "\n".join(data), []
                at = time.perf_counter()
                if output:
                    output.write(json.dumps({"at": at, "data": raw}) + "\n")
                    output.flush()
                if raw == "[DONE]":
                    trace.done = True
                    break
                event = json.loads(raw)
                if "error" in event:
                    raise RuntimeError(f"Streaming inference failed: {event['error']}")
                if event.get("id") is not None:
                    if trace.response_id is not None and trace.response_id != event["id"]:
                        raise ValueError("Response identity changed within one stream")
                    trace.response_id = event["id"]
                if event.get("usage") is not None:
                    trace.usage = event["usage"]
                for choice in event.get("choices", []):
                    if choice["index"] != 0:
                        raise ValueError("Unexpected multiple completion indices")
                    ids = choice.get("token_ids")
                    if ids is None:
                        raise ValueError("Engine omitted requested token IDs; text length is not a token count")
                    if ids:
                        if any(type(token) is not int or token < 0 for token in ids):
                            raise ValueError("Invalid emitted token ID")
                        trace.bursts.append((at, ids))
                    trace.text += choice.get("text", "")
                    if choice.get("finish_reason") is not None:
                        trace.finish_reason = choice["finish_reason"]
        trace.finished = time.perf_counter()
        trace.validate()
        if trace.response_id is None:
            raise ValueError("Stream omitted its response identity")
        return trace
    finally:
        if output:
            output.close()


def summarize_group(traces):
    if not traces:
        raise ValueError("No requests to measure")
    for trace in traces:
        trace.validate()
    wall = max(t.finished for t in traces) - min(t.submitted for t in traces)
    if wall <= 0:
        raise ValueError("Nonpositive group service interval")
    populated = [trace for trace in traces if trace.bursts]
    window = (max(t.bursts[-1][0] for t in populated) - min(t.bursts[0][0] for t in populated)
              if populated else 0.0)
    per_request = []
    for trace in traces:
        n = len(trace.token_ids)
        duration = trace.bursts[-1][0] - trace.bursts[0][0] if trace.bursts else 0.0
        gaps = sorted(b[0] - a[0] for a, b in zip(trace.bursts, trace.bursts[1:]))
        per_request.append({"tokens": n, "ttft_seconds": trace.bursts[0][0] - trace.submitted if trace.bursts else None,
            "service_seconds": trace.finished - trace.submitted,
            "arrival_decode_tokens_per_second": (n - 1) / duration if duration > 0 else None,
            "burst_gap_p95_seconds": gaps[max(0, math.ceil(len(gaps) * .95) - 1)] if gaps else None,
            "burst_gap_max_seconds": max(gaps) if gaps else None,
            "finish_reason": trace.finish_reason, "censored": trace.finish_reason == "length"})
    total = sum(len(t.token_ids) for t in traces)
    return {"requests": per_request, "total_output_tokens": total,
        "group_service_seconds": wall, "cold_service_tokens_per_second": total / wall,
        "group_arrival_decode_seconds": window,
        "aggregate_arrival_decode_tokens_per_second":
            sum(max(0, len(t.token_ids) - 1) for t in traces) / window if window > 0 else None,
        "scope": "HTTP arrival windows; includes interference; no native kernel throughput claim"}
