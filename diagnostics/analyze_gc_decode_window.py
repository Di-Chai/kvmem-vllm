"""Post-hoc interior decode diagnostic; never replaces the paired primary gate."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def interior(timeline, total, begin=32, end=480):
    if not 0 < begin < end < total:
        raise ValueError("The diagnostic window must exclude initial and terminal tokens")
    previous = float("-inf")
    points = []
    cumulative = 0
    # The qualifier stores each SSE packet's token count, not its cumulative count.
    for at, packet_tokens in timeline:
        if at < previous or type(packet_tokens) is not int or packet_tokens <= 0:
            raise ValueError("Invalid accepted-token packet timeline")
        previous = at
        cumulative += packet_tokens
        points.append((at, cumulative))
    if cumulative != total:
        raise ValueError("Timeline packet count differs from accepted output IDs")
    start = next(p for p in points if p[1] >= begin)
    finish = [p for p in points if p[1] <= end][-1]
    if not start[0] < finish[0] or not start[1] < finish[1]:
        raise ValueError("Insufficient interior token progress")
    return dict(started_at=start[0], finished_at=finish[0],
                first_tokens=start[1], last_tokens=finish[1],
                token_progress=finish[1]-start[1],
                seconds=finish[0]-start[0],
                throughput=(finish[1]-start[1])/(finish[0]-start[0]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    primary = read(args.run / "results.json")
    runs = read(args.run / "runs.json")
    if len(runs) != 10 or len(primary["comparisons"]) != 5:
        raise ValueError("Require the complete predeclared five-pair run")
    values = []
    evidence = {}
    for item in runs:
        pair, mode = item["pair"], item["mode"]
        directory = args.run / f"{pair:02d}-{mode}" / "client"
        result_path = directory / "measure-result.json"
        fixture_path = directory / "statistical-fixture.json"
        result, fixture = read(result_path), read(fixture_path)
        sample = interior(result["timeline"], len(result["token_ids"]))
        if mode == "delete":
            for kind in ("retired", "cpu_released"):
                events = [e for e in fixture["events"] if e["kind"] == kind]
                if not events or any(
                    not sample["started_at"] <= e["interval_begin"] < e["interval_end"] <= sample["finished_at"]
                    or not e["interval_end"] < e.get("following_decode_at", float("inf")) <= sample["finished_at"]
                    or e["after"]-e["before"] != 4 for e in events
                ):
                    raise ValueError("Interior diagnostic does not cover complete GC release witnesses")
        else:
            if any(e["kind"] in ("retired", "cpu_released") for e in fixture["events"]):
                raise ValueError("OFF fixture unexpectedly retired candidates")
        completions = [v for v in item["metrics"]["archive"]["completed"]
                       if v["request_id"].startswith(result["id"])]
        if len(completions) != 1:
            raise ValueError("Missing measured request terminal state")
        terminal = completions[0]
        sample.update(pair=pair, mode=mode, output_sha256=result["output_sha256"],
                      rejected_tokens=terminal["rejected_tokens"],
                      checkpoint_kind=terminal["checkpoint_kind"],
                      checkpoint_tokens=terminal["checkpoint_tokens"])
        values.append(sample)
        evidence.update({str(f):hashlib.sha256(f.read_bytes()).hexdigest()
                         for f in (result_path, fixture_path)})
    comparisons = []
    for pair in range(1, 6):
        off, delete = ({v["mode"]:v for v in values if v["pair"] == pair}[m]
                       for m in ("off", "delete"))
        if off["output_sha256"] != delete["output_sha256"]:
            raise ValueError("Paired output mismatch")
        comparisons.append(dict(pair=pair, off_throughput=off["throughput"],
                                delete_throughput=delete["throughput"],
                                throughput_loss=1-delete["throughput"]/off["throughput"]))
    losses = [v["throughput_loss"] for v in comparisons]
    report = dict(
        scope="Exploratory post-hoc SSE throughput within accepted tokens32..480, excluding terminal checkpoints",
        qualification=False,
        rationale="OFF saves terminal M; DELETE defers M to an earlier P. This diagnostic investigates that endpoint difference.",
        primary_status=primary["status"],
        primary_result_sha256=hashlib.sha256((args.run/"results.json").read_bytes()).hexdigest(),
        requested_window=[32,480], samples=values, comparisons=comparisons,
        median_loss=statistics.median(losses), minimum_loss=min(losses), maximum_loss=max(losses),
        limitations="Window selected after observing early pairs; not a replacement noninferiority test, native GPU-step trace, or semantic-quality test.",
        evidence_sha256=evidence)
    with args.output.open("x") as f:
        f.write(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("primary_status","median_loss","minimum_loss","maximum_loss")}))


if __name__ == "__main__":
    main()
