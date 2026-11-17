#!/usr/bin/env python3
"""Mini Project 4.2 - multi-rate environmental monitoring pipeline with timing-compliance reporting.

One gateway, ONE scheduling loop, several sensor streams with different required sampling periods.

Scheduling loop (single processor, non-preemptive: a read in progress cannot be interrupted)
    1. at time t, every stream whose next required release r <= t is DUE.  A release that has already been
       superseded by the next one (r + P <= t) was never serviced in time and is counted MISSED and dropped.
    2. exactly ONE due stream is serviced per pass, chosen by the rule under test:
         FP : fixed priority, shortest required period first (rate-monotonic priority; ties by name)
         EDF: earliest deadline first, deadline = release + 10 % of the period (the latest start that still counts as on-time)
         RR : round-robin, the due stream that follows the last serviced one in a fixed cyclic order
    3. servicing takes a random processing time (short, with an occasional long tail); the clock advances by it.
       If nothing is due, the clock jumps to the earliest next release.
Classification of every required sample (lateness L = service start - required release):
    ON-TIME  L <= 10 % of the stream's period     LATE  10 % < L < P (still delivered before the next release)     MISSED  not serviced before the next release
Alerts: after each serviced sample the rule "value crossed the threshold from below since the previous sample" is evaluated.
    Detection time = end of the read (service start + processing time);  true crossing time = root of the continuous signal.

Usage:
    python env_pipeline.py --all                  six configurations (3 rules x 2 loads), reports and figures
    python env_pipeline.py --rule RR --streams 4  one configuration
"""
import argparse
import heapq
import math
import os
import random
import statistics as st
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DURATION = 300.0
ON_TIME_FRAC = 0.10


@dataclass(eq=False)
class Stream:
    name: str
    unit: str
    period: float                 # required sampling period (s)
    signal: callable              # true continuous signal
    threshold: float
    proc_mean: float              # typical processing time of one read (s)
    tail_prob: float = 0.05       # probability that a read overruns
    tail_mult: tuple = (4.0, 10.0)  # overrun = proc_mean * uniform(tail_mult)
    # running state / counters
    next_rel: float = 0.0
    k: int = 0
    on_time: int = 0
    late: int = 0
    missed: int = 0
    lateness: list = field(default_factory=list)       # L of every serviced sample
    samples: list = field(default_factory=list)        # (service start, value, class, release)
    miss_times: list = field(default_factory=list)
    alerts: list = field(default_factory=list)         # (detect time, true crossing time)
    prev: tuple = None                                 # (t, value) of the previous serviced sample

    @property
    def required(self):
        return self.on_time + self.late + self.missed


def make_streams(n):
    s = [
        Stream("temperature", "C", 1.0, lambda t: 22 + 5 * math.sin(2 * math.pi * t / 30) + 0.4 * math.sin(2 * math.pi * 0.37 * t), 25.0, 0.040),
        Stream("humidity", "%", 0.5, lambda t: 55 + 15 * math.sin(2 * math.pi * t / 20 + 0.5) + 1.0 * math.sin(2 * math.pi * 0.61 * t), 65.0, 0.030),
        Stream("air_quality", "ppm", 2.0, lambda t: 600 + 300 * math.sin(2 * math.pi * t / 40 + 1.0) + 25 * math.sin(2 * math.pi * 0.23 * t), 800.0, 0.120),
    ]
    if n == 4:
        s.append(Stream("vibration", "mm/s", 0.1, lambda t: 3 + 2.5 * math.sin(2 * math.pi * t / 15) + 0.2 * math.sin(2 * math.pi * 1.3 * t), 5.0, 0.025))
    return s


def true_crossing(sig, th, t0, t1):
    """First rising crossing of the continuous signal in (t0, t1], or None (scan + bisection)."""
    n = max(20, int((t1 - t0) / 0.005))
    prev_t, prev_v = t0, sig(t0)
    for i in range(1, n + 1):
        t = t0 + (t1 - t0) * i / n
        v = sig(t)
        if prev_v < th <= v:
            lo, hi = prev_t, t
            for _ in range(30):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if sig(mid) < th else (lo, mid)
            return hi
        prev_t, prev_v = t, v
    return None


def all_true_crossings(sig, th, duration):
    out, t, step = [], 0.0, 0.5
    while t < duration:
        c = true_crossing(sig, th, t, min(t + step, duration))
        if c is not None:
            out.append(c)
        t += step
    return out


def proc_time(s, rnd):
    if rnd.random() < s.tail_prob:
        return s.proc_mean * rnd.uniform(*s.tail_mult)
    return s.proc_mean * rnd.uniform(0.7, 1.3)


def pick(rule, due, last, order):
    if rule == "FP":
        return min(due, key=lambda s: (s.period, s.name))
    if rule == "EDF":
        return min(due, key=lambda s: (s.next_rel + ON_TIME_FRAC * s.period, s.name))
    if rule == "RR":
        start = order.index(last) + 1 if last in order else 0
        for i in range(len(order)):
            cand = order[(start + i) % len(order)]
            if cand in due:
                return cand
    raise ValueError(rule)


def run(rule, n_streams, seed, duration=DURATION):
    rnd = random.Random(seed)
    streams = make_streams(n_streams)
    heap = [(s.next_rel, i) for i, s in enumerate(streams)]
    heapq.heapify(heap)
    t, last, busy = 0.0, None, 0.0
    while True:
        # drop superseded releases (missed), collect due streams
        due = []
        for s in streams:
            while s.next_rel + s.period <= t and s.next_rel < duration:
                s.missed += 1; s.miss_times.append(s.next_rel); s.next_rel += s.period
            if s.next_rel <= t and s.next_rel < duration:
                due.append(s)
        if not due:
            nxt = [s.next_rel for s in streams if s.next_rel < duration]
            if not nxt:
                break
            t = max(t, min(nxt))
            continue
        s = pick(rule, due, last, streams)
        L = t - s.next_rel
        cls = "on_time" if L <= ON_TIME_FRAC * s.period + 1e-12 else "late"
        x = s.signal(t)
        c = proc_time(s, rnd)
        detect = t + c
        if cls == "on_time":
            s.on_time += 1
        else:
            s.late += 1
        s.lateness.append(L)
        s.samples.append((t, x, cls, s.next_rel))
        if s.prev is not None and s.prev[1] < s.threshold <= x:
            tc = true_crossing(s.signal, s.threshold, s.prev[0], t)
            s.alerts.append((detect, tc if tc is not None else t))
        s.prev = (t, x)
        s.next_rel += s.period
        busy += c
        last = s
        t = detect
        if s.name and t > duration * 1.5:
            break
    # tidy: releases never reached before the run ended count as missed only if their window closed inside the run
    return dict(rule=rule, n=n_streams, seed=seed, streams=streams, busy=busy, end=t, duration=duration)


# ---------------------------------------------------------------- reporting
def report(res):
    d = res["duration"]
    util = sum(s.proc_mean / s.period for s in res["streams"])
    lines = [f"Configuration: rule {res['rule']}, {res['n']} streams, seed {res['seed']}, run length {d:g} s; "
             f"nominal utilisation (mean read time / period) = {util:.2f}, measured busy fraction = {res['busy'] / d:.2f}"]
    lines.append(f"  {'stream':<12}{'period':>7}{'required':>9}{'on-time':>9}{'late':>6}{'missed':>8}{'%on-time':>9}{'mean ovr':>10}{'max ovr':>9}"
                 f"{'crossings':>10}{'detected':>9}{'mean alert':>11}{'max alert':>10}")
    lines.append(f"  {'':<12}{'(s)':>7}{'':>9}{'':>9}{'':>6}{'':>8}{'':>9}{'(ms)':>10}{'(ms)':>9}{'(true)':>10}{'':>9}{'lat (ms)':>11}{'lat (ms)':>10}")
    rows = []
    for s in res["streams"]:
        req = s.required
        tcs = all_true_crossings(s.signal, s.threshold, d)
        lat = [(a - b) * 1000 for a, b in s.alerts if b is not None and b < d]
        det = len(lat)
        mo = st.mean(s.lateness) * 1000 if s.lateness else float("nan")
        xo = max(s.lateness) * 1000 if s.lateness else float("nan")
        lines.append(f"  {s.name:<12}{s.period:>7.2f}{req:>9}{s.on_time:>9}{s.late:>6}{s.missed:>8}{100 * s.on_time / req:>8.1f}%{mo:>10.1f}{xo:>9.1f}"
                     f"{len(tcs):>10}{det:>9}{(st.mean(lat) if lat else float('nan')):>11.1f}{(max(lat) if lat else float('nan')):>10.1f}")
        rows.append(dict(name=s.name, req=req, on=s.on_time, late=s.late, missed=s.missed, crossings=len(tcs), detected=det,
                         mean_ovr=mo, max_ovr=xo, mean_alert=(st.mean(lat) if lat else float("nan")), max_alert=(max(lat) if lat else float("nan"))))
    return "\n".join(lines), rows


def plots(res, tag):
    streams = res["streams"]
    W = 45.0
    fig, axes = plt.subplots(len(streams), 1, figsize=(11, 2.1 * len(streams) + 0.8), sharex=True)
    for ax, s in zip(axes, streams):
        tt = [i * 0.05 for i in range(int(W / 0.05))]
        ax.plot(tt, [s.signal(t) for t in tt], color="0.65", lw=1.2, label="true signal")
        ax.axhline(s.threshold, color="tab:green", ls="--", lw=1, label="threshold")
        for cls, mk, col, lab in (("on_time", ".", "tab:blue", "on-time"), ("late", "^", "tab:orange", "late")):
            pts = [(a, b) for a, b, c, _ in s.samples if c == cls and a < W]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], mk, color=col, ms=4 if cls == "late" else 3, label=lab, ls="none")
        ms = [t for t in s.miss_times if t < W]
        if ms:
            ax.plot(ms, [s.signal(t) for t in ms], "x", color="red", ms=5, ls="none", label="missed")
        al = [(d, tc) for d, tc in s.alerts if d < W]
        for d, tc in al:
            ax.axvline(d, color="tab:purple", lw=0.8, alpha=0.7)
        ax.set_ylabel(f"{s.name}\n({s.unit}, P={s.period:g} s)", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=5, fontsize=7, loc="upper right")
    axes[0].set_title(f"Signals and serviced samples - rule {res['rule']}, {res['n']} streams (first {W:g} s; purple lines = alert detections)", fontsize=10)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, f"signals_{tag}.png"), dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    names = [f"{s.name}\n(n = {s.required})" for s in streams]
    pct = lambda v, s: 100.0 * v / s.required
    on = [pct(s.on_time, s) for s in streams]; la = [pct(s.late, s) for s in streams]; mi = [pct(s.missed, s) for s in streams]
    ax.bar(names, on, color="tab:blue", label="on-time")
    ax.bar(names, la, bottom=on, color="tab:orange", label="late")
    ax.bar(names, mi, bottom=[x + y for x, y in zip(on, la)], color="tab:red", label="missed")
    for i, s in enumerate(streams):
        ax.text(i, 101, f"{s.on_time}/{s.late}/{s.missed}", ha="center", fontsize=8)
    ax.set_ylabel("% of required samples"); ax.set_ylim(0, 112)
    ax.set_title(f"On-time / late / missed - rule {res['rule']}, {res['n']} streams (labels: on-time/late/missed counts)", fontsize=9)
    ax.legend(loc="lower right", fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, f"compliance_{tag}.png"), dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rule", choices=["FP", "EDF", "RR"], default="FP")
    ap.add_argument("--streams", type=int, choices=[3, 4], default=3)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if not a.all:
        res = run(a.rule, a.streams, a.seed)
        print(report(res)[0]); plots(res, f"{a.rule}_{a.streams}")
        return
    out, summary = [], {}
    P = lambda s="": (print(s), out.append(s))
    P("Mini Project 4.2 - per-stream timing-compliance reports (seed 7). ovr = lateness of the service start after the required release.")
    P()
    for n in (3, 4):
        for rule in ("FP", "EDF", "RR"):
            res = run(rule, n, 7)
            txt, rows = report(res)
            P(txt); P()
            plots(res, f"{rule}_{n}")
    # multi-seed comparison
    P("Average over 20 seeds: % of required samples on-time / late / missed per stream")
    P(f"{'load':<10}{'rule':<6}" + "".join(f"{nm:>26}" for nm in ("temperature", "humidity", "air_quality", "vibration")))
    for n in (3, 4):
        for rule in ("FP", "EDF", "RR"):
            acc = {}
            for sd in range(1, 21):
                res = run(rule, n, sd)
                for s in res["streams"]:
                    r = s.required
                    acc.setdefault(s.name, []).append((100 * s.on_time / r, 100 * s.late / r, 100 * s.missed / r))
            cells = []
            for nm in ("temperature", "humidity", "air_quality", "vibration"):
                if nm in acc:
                    v = [st.mean(x[i] for x in acc[nm]) for i in range(3)]
                    cells.append(f"{v[0]:>8.1f} /{v[1]:>6.1f} /{v[2]:>6.1f}")
                else:
                    cells.append("-")
            P(f"{n} streams  {rule:<6}" + "".join(f"{c:>26}" for c in cells))
    open(os.path.join(HERE, "pipeline_output.txt"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
