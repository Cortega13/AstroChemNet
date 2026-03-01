"""Analyze a PyTorch profiler chrome trace JSON.

This script is tailored for traces created via
[`torch.profiler.profile()`](src/trainer.py:333) and exported with
[`export_chrome_trace()`](src/trainer.py:350).

It prints:
- header/device metadata
- event/category totals (inclusive time; *double-counts* when events nest)
- per-thread totals
- exclusive/self-time breakdown for the busiest CPU thread
- CUDA runtime call summaries, kernel summaries, and GPU memcpy summaries

The trace file can be large; this script loads it fully into memory.
The included trace in this repo (~20MB, ~80k events) is fine for that.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Event:
    ph: str
    cat: str
    name: str
    pid: Any
    tid: Any
    ts: float
    dur: float

    @property
    def end(self) -> float:
        return self.ts + self.dur


def _as_float(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) else None


def iter_x_events(trace_events: Iterable[dict[str, Any]]) -> Iterable[Event]:
    """Yield only complete events (`ph == 'X'`) with numeric `ts`/`dur`."""
    for ev in trace_events:
        if ev.get("ph") != "X":
            continue
        ts = _as_float(ev.get("ts"))
        dur = _as_float(ev.get("dur"))
        if ts is None or dur is None:
            continue
        yield Event(
            ph="X",
            cat=str(ev.get("cat", "")),
            name=str(ev.get("name", "")),
            pid=ev.get("pid"),
            tid=ev.get("tid"),
            ts=ts,
            dur=dur,
        )


def compute_wall_span_us(events: Iterable[Event]) -> tuple[float, float, float]:
    """Return (min_ts, max_end, span) in microseconds."""
    min_ts = float("inf")
    max_end = float("-inf")
    for ev in events:
        if ev.ts < min_ts:
            min_ts = ev.ts
        if ev.end > max_end:
            max_end = ev.end
    if min_ts == float("inf") or max_end == float("-inf"):
        return (0.0, 0.0, 0.0)
    return (min_ts, max_end, max_end - min_ts)


def topk(mapping: dict[Any, float], k: int) -> list[tuple[Any, float]]:
    return sorted(mapping.items(), key=lambda kv: kv[1], reverse=True)[:k]


def inclusive_totals(
    events: Iterable[Event],
) -> tuple[Counter, dict[str, float], Counter, dict[str, float]]:
    """(cat_counts, cat_dur, name_counts, name_dur), inclusive sums."""
    cat_counts: Counter[str] = Counter()
    name_counts: Counter[str] = Counter()
    cat_dur: dict[str, float] = defaultdict(float)
    name_dur: dict[str, float] = defaultdict(float)
    for ev in events:
        cat_counts[ev.cat] += 1
        name_counts[ev.name] += 1
        cat_dur[ev.cat] += ev.dur
        name_dur[ev.name] += ev.dur
    return cat_counts, cat_dur, name_counts, name_dur


def per_thread_totals(
    events: Iterable[Event],
) -> tuple[Counter, dict[tuple[Any, Any], float]]:
    """Return (counts_by_(pid,tid), dur_by_(pid,tid))."""
    cnt: Counter[tuple[Any, Any]] = Counter()
    dur: dict[tuple[Any, Any], float] = defaultdict(float)
    for ev in events:
        key = (ev.pid, ev.tid)
        cnt[key] += 1
        dur[key] += ev.dur
    return cnt, dur


def pick_busiest_cpu_thread(events: Iterable[Event]) -> tuple[Any, Any] | None:
    """Heuristic: (pid,tid) with the most `cat == cpu_op` events."""
    cpu_counts: Counter[tuple[Any, Any]] = Counter()
    for ev in events:
        if ev.cat == "cpu_op":
            cpu_counts[(ev.pid, ev.tid)] += 1
    return cpu_counts.most_common(1)[0][0] if cpu_counts else None


def exclusive_self_time_for_thread(
    thread_events: list[Event],
) -> dict[str, float]:
    """Compute per-name *exclusive* (self) time for one thread.

    Notes:
    - This assumes well-nested intervals on a single thread.
    - If the trace has overlapping events on the same tid, the result may be off.
    """
    # Sort by start, and for identical starts put longer events first so nesting works.
    evs = sorted(thread_events, key=lambda e: (e.ts, -e.dur))
    stack: list[tuple[Event, float]] = []  # (event, accumulated_child_time)
    self_time_by_name: dict[str, float] = defaultdict(float)

    for ev in evs:
        # Close any events that ended before this start.
        while stack and ev.ts >= stack[-1][0].end:
            done, child = stack.pop()
            self_time_by_name[done.name] += max(0.0, done.dur - child)
            if stack:
                parent_ev, parent_child = stack[-1]
                stack[-1] = (parent_ev, parent_child + done.dur)

        stack.append((ev, 0.0))

    # Close any remaining events.
    while stack:
        done, child = stack.pop()
        self_time_by_name[done.name] += max(0.0, done.dur - child)
        if stack:
            parent_ev, parent_child = stack[-1]
            stack[-1] = (parent_ev, parent_child + done.dur)

    return self_time_by_name


def summarize_memcpy(trace_events: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Return dict copyKind -> {count, dur_us, bytes}."""
    out: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "dur_us": 0.0, "bytes": 0.0}
    )
    for ev in trace_events:
        if ev.get("ph") != "X" or ev.get("cat") != "gpu_memcpy":
            continue
        dur = _as_float(ev.get("dur"))
        if dur is None:
            continue
        args = ev.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        kind = str(args.get("copyKind") or args.get("kind") or "unknown")
        b = args.get("bytes") or args.get("numBytes") or 0
        b = float(b) if isinstance(b, (int, float)) else 0.0
        out[kind]["count"] += 1.0
        out[kind]["dur_us"] += dur
        out[kind]["bytes"] += b
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "trace",
        nargs="?",
        default="scripts/profile_analysis/emulator_profile_trace.json",
        help="Path to chrome trace JSON exported by torch profiler.",
    )
    ap.add_argument(
        "--top", type=int, default=25, help="How many rows to show for top-N tables."
    )
    args = ap.parse_args()

    trace_path = args.trace
    print(f"Trace: {trace_path}")
    if os.path.exists(trace_path):
        print(f"Size: {os.path.getsize(trace_path)} bytes")

    with open(trace_path, "r") as f:
        data = json.load(f)

    print("\n== Header ==")
    for k in [
        "schemaVersion",
        "cupti_version",
        "cuda_runtime_version",
        "cuda_driver_version",
        "trace_id",
    ]:
        if k in data:
            print(f"{k}: {data[k]}")

    devs = data.get("deviceProperties") or []
    if devs:
        d0 = devs[0]
        if isinstance(d0, dict):
            print("device[0].name:", d0.get("name"))
            print("device[0].totalGlobalMem:", d0.get("totalGlobalMem"))
            print(
                "device[0].compute:",
                f"{d0.get('computeMajor')}.{d0.get('computeMinor')}",
            )
            print("device[0].numSms:", d0.get("numSms"))

    trace_events: list[dict[str, Any]] = data.get("traceEvents") or []
    print("\ntraceEvents:", len(trace_events))
    x_events = list(iter_x_events(trace_events))
    print("X events (complete events with ts/dur):", len(x_events))

    min_ts, max_end, span = compute_wall_span_us(x_events)
    print(f"Wall span (us): {span:.3f}  [min_ts={min_ts:.3f}, max_end={max_end:.3f}]")

    cat_counts, cat_dur, name_counts, name_dur = inclusive_totals(x_events)

    print("\n== Inclusive totals by category (us) ==")
    for cat, tot in topk(cat_dur, args.top):
        print(f"{cat:18s} {tot:12.3f} (count={cat_counts[cat]})")

    print("\n== Inclusive totals by name (us) ==")
    for name, tot in topk(name_dur, args.top):
        print(f"{name:65s} {tot:12.3f} (count={name_counts[name]})")

    print(
        "\nNOTE: inclusive totals double-count nested events. Use the exclusive/self-time section for one thread."
    )

    # Per-thread totals (inclusive)
    cnt_by_tid, dur_by_tid = per_thread_totals(x_events)
    print("\n== Top threads by inclusive total duration (us) ==")
    for (pid, tid), tot in sorted(
        dur_by_tid.items(), key=lambda kv: kv[1], reverse=True
    )[: args.top]:
        print(
            f"pid={pid!s:10s} tid={tid!s:10s} dur_us={tot:12.3f} events={cnt_by_tid[(pid, tid)]}"
        )

    busiest = pick_busiest_cpu_thread(x_events)
    if busiest is not None:
        bpid, btid = busiest
        thread_events = [e for e in x_events if (e.pid, e.tid) == busiest]
        print("\n== Busiest CPU thread (by cpu_op count) ==")
        cpu_op_count = sum(1 for e in thread_events if e.cat == "cpu_op")
        print(
            f"pid={bpid} tid={btid} cpu_op_events={cpu_op_count} total_events={len(thread_events)}"
        )
        st = exclusive_self_time_for_thread(thread_events)
        print("\n== Exclusive/self time by name for busiest thread (us) ==")
        for name, tot in sorted(st.items(), key=lambda kv: kv[1], reverse=True)[
            : args.top
        ]:
            print(f"{name:65s} {tot:12.3f}")

    # Kernel summary
    kernels = [e for e in x_events if e.cat == "kernel"]
    if kernels:
        k_cnt: Counter[str] = Counter()
        k_dur: dict[str, float] = defaultdict(float)
        k_max: dict[str, float] = defaultdict(float)
        for e in kernels:
            k_cnt[e.name] += 1
            k_dur[e.name] += e.dur
            if e.dur > k_max[e.name]:
                k_max[e.name] = e.dur

        print("\n== Kernel summary ==")
        print("unique kernels:", len(k_cnt), "launches:", sum(k_cnt.values()))
        print("\nTop kernels by total duration (us):")
        for name, tot in sorted(k_dur.items(), key=lambda kv: kv[1], reverse=True)[
            : args.top
        ]:
            print(
                f"{name[:90]:90s} {tot:10.3f} (count={k_cnt[name]}, max={k_max[name]:.3f})"
            )

        print("\nTop kernels by launch count:")
        for name, cnt in k_cnt.most_common(args.top):
            print(f"{name[:90]:90s} {cnt:6d} (total_dur={k_dur[name]:.3f})")

    # CUDA runtime summary
    rt = [e for e in x_events if e.cat == "cuda_runtime"]
    if rt:
        rt_cnt: Counter[str] = Counter()
        rt_dur: dict[str, float] = defaultdict(float)
        for e in rt:
            rt_cnt[e.name] += 1
            rt_dur[e.name] += e.dur

        print("\n== CUDA runtime calls by total duration (us) ==")
        for name, tot in sorted(rt_dur.items(), key=lambda kv: kv[1], reverse=True)[
            : args.top
        ]:
            print(f"{name:30s} {tot:12.3f} (count={rt_cnt[name]})")

    # GPU memcpy summary
    memcpy = summarize_memcpy(trace_events)
    if memcpy:
        print("\n== GPU memcpy by kind ==")
        for kind, info in sorted(
            memcpy.items(), key=lambda kv: kv[1]["dur_us"], reverse=True
        ):
            mb = info["bytes"] / (1024 * 1024)
            print(
                f"{kind:12s} dur_us={info['dur_us']:10.3f} count={int(info['count']):5d} bytes={mb:10.3f} MiB"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
