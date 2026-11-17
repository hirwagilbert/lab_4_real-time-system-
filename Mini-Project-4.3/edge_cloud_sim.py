#!/usr/bin/env python3
"""Mini Project 4.3 - edge-versus-cloud latency simulator for a real-time IoT decision task.

A sensor produces one reading every 100 ms (10 Hz).  Each reading needs one "decide" computation (anomaly: yes/no) and the
decision must be available within DEADLINE = 100 ms of the reading.  20 % of readings are "hard" (ambiguous) and need a heavier model.

EDGE   (all latencies in ms):  transport ~0.5 ms; compute is a low-variance lognormal on limited hardware:
          easy  median 25 ms (sigma 0.15)      hard  median 70 ms (sigma 0.20)
CLOUD  = upload leg + remote compute + download leg
          legs   : lognormal body (upload median 18 ms, download 12 ms, sigma 0.25) multiplied by (1 + congestion_level), plus a
                   congestion SPIKE with probability  p = 0.005 + 0.15*congestion_level  per leg, size  30 ms * Pareto(alpha = 1.6)
                   (heavy tail: infinite variance, occasional spikes of hundreds of ms)
          compute: easy median 6 ms, hard median 12 ms (sigma 0.2) on a powerful server
HYBRID : a 5 ms edge pre-check classifies the reading.  EASY -> finished on the edge.  HARD -> sent to the cloud.
HEDGED : like HYBRID, but a hard reading is ALSO computed on the edge in parallel; the first answer wins (costs edge energy).

All paths face the same event sequence: every event owns its own pre-drawn random numbers (common random numbers), so the
same event is evaluated by every architecture and a congestion sweep changes only the cloud's network behaviour.

Usage:  python edge_cloud_sim.py --all            baseline table, sweep, figures (writes results_output.txt, results_table.csv, fig*.png)
        python edge_cloud_sim.py --congestion 0.5 --deadline 80 --events 5000
"""
import argparse
import csv
import math
import os
import random
import statistics as st
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

DEADLINE = 100.0
BASELINE_CONGESTION = 0.3
P_HARD = 0.20
PRECHECK = 5.0


@dataclass
class Event:
    hard: bool
    z_edge: float
    z_remote: float
    z_up: float
    z_down: float
    u_up: float
    u_down: float
    pareto_up: float
    pareto_down: float


def make_events(n, seed=1):
    out = []
    for i in range(n):
        r = random.Random(seed * 1_000_003 + i)
        out.append(Event(hard=r.random() < P_HARD, z_edge=r.gauss(0, 1), z_remote=r.gauss(0, 1), z_up=r.gauss(0, 1), z_down=r.gauss(0, 1),
                         u_up=r.random(), u_down=r.random(), pareto_up=1 - r.random(), pareto_down=1 - r.random()))
    return out


def lognormal(median, sigma, z):
    return median * math.exp(sigma * z)


def edge_compute(ev):
    return lognormal(70.0, 0.20, ev.z_edge) if ev.hard else lognormal(25.0, 0.15, ev.z_edge)


def edge_latency(ev):
    """End-to-end latency of the edge path (ms): negligible transport + local compute."""
    return 0.5 + edge_compute(ev)


def leg(median, z, u, pareto_u, congestion):
    body = lognormal(median, 0.25, z) * (1.0 + congestion)
    spike = 30.0 * pareto_u ** (-1.0 / 1.6) if u < 0.005 + 0.15 * congestion else 0.0
    return body + spike


def cloud_latency(ev, congestion):
    """End-to-end latency of the cloud path (ms) at a congestion level in [0, 1]."""
    up = leg(18.0, ev.z_up, ev.u_up, ev.pareto_up, congestion)
    down = leg(12.0, ev.z_down, ev.u_down, ev.pareto_down, congestion)
    remote = lognormal(12.0 if ev.hard else 6.0, 0.20, ev.z_remote)
    return up + remote + down


def hybrid_latency(ev, congestion):
    if not ev.hard:
        return PRECHECK + edge_latency(ev)
    return PRECHECK + cloud_latency(ev, congestion)


def hedged_latency(ev, congestion):
    if not ev.hard:
        return PRECHECK + edge_latency(ev)
    return PRECHECK + min(cloud_latency(ev, congestion), edge_latency(ev))


def pct(sorted_vals, q):
    """Nearest-rank percentile of an ascending list."""
    k = max(1, math.ceil(q / 100.0 * len(sorted_vals)))
    return sorted_vals[k - 1]


def summarise(lat, deadline):
    s = sorted(lat)
    return dict(mean=st.mean(lat), p95=pct(s, 95), p99=pct(s, 99), worst=s[-1],
                comp=100.0 * sum(1 for x in lat if x <= deadline) / len(lat), missed=sum(1 for x in lat if x > deadline))


def paths(events, congestion):
    return {"edge": [edge_latency(e) for e in events],
            "cloud": [cloud_latency(e, congestion) for e in events],
            "hybrid": [hybrid_latency(e, congestion) for e in events],
            "hedged": [hedged_latency(e, congestion) for e in events]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--events", type=int, default=20000)
    ap.add_argument("--congestion", type=float, default=BASELINE_CONGESTION)
    ap.add_argument("--deadline", type=float, default=DEADLINE)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    events = make_events(a.events, a.seed)
    P_ = paths(events, a.congestion)
    lines = []
    P = lambda s="": (print(s), lines.append(s))

    def table(cong, deadline, label):
        L = paths(events, cong)
        P(f"{label}: congestion {cong:.2f}, deadline {deadline:g} ms, {len(events)} events ({sum(e.hard for e in events)} hard)")
        P(f"  {'architecture':<14}{'mean':>8}{'p95':>8}{'p99':>8}{'worst':>9}{'compliance':>12}{'missed':>8}")
        rows = []
        for name in ("edge", "cloud", "hybrid", "hedged"):
            r = summarise(L[name], deadline)
            rows.append((name, r))
            P(f"  {name:<14}{r['mean']:>8.1f}{r['p95']:>8.1f}{r['p99']:>8.1f}{r['worst']:>9.1f}{r['comp']:>11.2f}%{r['missed']:>8}")
        return rows

    if not a.all:
        table(a.congestion, a.deadline, "single run")
        return

    P("Mini Project 4.3 - edge versus cloud (all latencies in ms)")
    P()
    base = table(BASELINE_CONGESTION, DEADLINE, "BASELINE")
    P()
    with open(os.path.join(HERE, "results_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["architecture", "congestion", "mean_ms", "p95_ms", "p99_ms", "worst_ms", "compliance_pct"])
        for cong in (BASELINE_CONGESTION,):
            for name, r in base:
                w.writerow([name, cong, round(r["mean"], 2), round(r["p95"], 2), round(r["p99"], 2), round(r["worst"], 2), round(r["comp"], 3)])
    for cong in (0.0, 0.6, 0.9):
        table(cong, DEADLINE, f"CONGESTION {cong}")
        P()

    P("Deadline sensitivity at baseline congestion (compliance %):")
    P(f"  {'deadline':>9}" + "".join(f"{n:>9}" for n in ("edge", "cloud", "hybrid", "hedged")))
    L = paths(events, BASELINE_CONGESTION)
    for d in (50, 75, 100, 150, 200):
        P(f"  {d:>7} ms" + "".join(f"{summarise(L[n], d)['comp']:>8.2f}%" for n in ("edge", "cloud", "hybrid", "hedged")))
    P()

    # ---- congestion sweep
    levels = [i / 50 for i in range(0, 51)]          # 0.00 .. 1.00 step 0.02
    sweep = {n: [] for n in ("edge", "cloud", "hybrid", "hedged")}
    for c in levels:
        L = paths(events, c)
        for n in sweep:
            sweep[n].append(summarise(L[n], DEADLINE)["comp"])
    P("Compliance versus congestion (deadline 100 ms):")
    P(f"  {'congestion':>10}" + "".join(f"{n:>9}" for n in sweep))
    for i, c in enumerate(levels):
        if round(c * 10, 6) == int(round(c * 10)) and (i % 5 == 0):
            P(f"  {c:>10.1f}" + "".join(f"{sweep[n][i]:>8.2f}%" for n in sweep))
    for target in (99.9, 99.0, 95.0):
        ok = [c for c, v in zip(levels, sweep["cloud"]) if v >= target]
        P(f"  cloud stays >= {target}% compliant up to congestion {max(ok) if ok else None}" if ok else f"  cloud never reaches {target}% (even at congestion 0: {sweep['cloud'][0]:.2f}%)")
    P(f"  edge compliance is {sweep['edge'][0]:.2f}% at every congestion level (the edge path does not use the network)")
    open(os.path.join(HERE, "results_output.txt"), "w").write("\n".join(lines) + "\n")

    # ---- figure 1: overlaid latency distributions
    L = paths(events, BASELINE_CONGESTION)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    bins = [i * 4 for i in range(0, 76)]
    for n, col in (("edge", "tab:blue"), ("cloud", "tab:red"), ("hybrid", "tab:green")):
        a1.hist([min(x, 299.9) for x in L[n]], bins=bins, alpha=0.45, color=col, label=n, density=True)
    a1.axvline(DEADLINE, color="black", ls="--", lw=1.4, label=f"deadline {DEADLINE:g} ms")
    a1.set_xlabel("end-to-end latency (ms; values above 300 ms are shown in the last bin)"); a1.set_ylabel("density")
    a1.set_title(f"Figure 1a - latency histograms, congestion {BASELINE_CONGESTION}", fontsize=10); a1.legend(); a1.grid(alpha=0.3)
    for n, col in (("edge", "tab:blue"), ("cloud", "tab:red"), ("hybrid", "tab:green"), ("hedged", "tab:purple")):
        s = sorted(L[n]); m = len(s)
        a2.step(s, [1 - (i + 1) / m + 1e-5 for i in range(m)], color=col, label=n, where="post")
    a2.axvline(DEADLINE, color="black", ls="--", lw=1.4)
    a2.set_yscale("log"); a2.set_xscale("log"); a2.set_xlim(10, 2000); a2.set_ylim(5e-5, 1.2)
    a2.set_xlabel("latency (ms, log scale)"); a2.set_ylabel("P(latency > x)  (log scale)")
    a2.set_title("Figure 1b - tail of the distributions (complementary CDF)", fontsize=10); a2.legend(); a2.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig1_latency_distributions.png"), dpi=150); plt.close(fig)

    # ---- figure 2: compliance versus congestion
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(levels, sweep["cloud"], color="tab:red", lw=2, label="cloud")
    ax.plot(levels, sweep["hybrid"], color="tab:green", lw=2, label="hybrid (hard -> cloud)")
    ax.plot(levels, sweep["hedged"], color="tab:purple", lw=2, ls="-.", label="hybrid + edge hedge")
    ax.axhline(sweep["edge"][0], color="tab:blue", lw=2, ls="--", label="edge (flat reference)")
    ax.axhline(99.0, color="0.6", lw=0.8, ls=":"); ax.text(1.0, 99.05, "99 %", ha="right", va="bottom", fontsize=8, color="0.4")
    ax.set_xlabel("congestion level"); ax.set_ylabel(f"deadline compliance (%), deadline {DEADLINE:g} ms")
    ax.set_title("Figure 2 - deadline compliance versus congestion", fontsize=10)
    ax.set_ylim(min(sweep["cloud"]) - 2, 100.5); ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig2_compliance_vs_congestion.png"), dpi=150); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
