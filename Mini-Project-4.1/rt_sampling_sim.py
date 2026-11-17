#!/usr/bin/env python3
"""Mini Project 4.1 - sampling jitter and processing overruns in a fixed-period sensing-and-control loop.

Model (Section 4.1 timing budget, Section 4.5 scan cycle)
  * the physical process is true_signal(t): a sine plus small continuous (deterministic) ripple
  * IDEAL loop      : sample k is taken at t_k = k*T and acted on at once (zero jitter, zero compute time)
  * REAL loop       : slot k is nominally at k*T.  The sample is taken at
                          s_k = max(k*T + jitter_k, f_{k-1})        (a scan cannot start before the previous one ended)
                      and the cycle finishes at f_k = s_k + c_k, where c_k is the computation time:
                          usually uniform(c_lo, c_hi) (short), with probability p drawn from the overrun range (> T).
                      An overrun therefore delays every following sample until the loop catches up (like a PLC scan overrun).
  * threshold controller : issues an actuation command when a sample crosses the threshold from below.
                      ideal -> at t_k ;  real -> at f_k (the command is issued after the computation).
  * P controller         : u_k = Kp*(setpoint - x_k), held until the next command (zero-order hold).

Usage
    python rt_sampling_sim.py --sweep                 full sweep, tables, figures (writes sweep_output.txt, sweep_table.csv, fig*.png)
    python rt_sampling_sim.py --jitter 0.3 --overrun-prob 0.05          one run, prints its statistics
Time is in seconds; jitter bound and overrun sizes are given as fractions/multiples of T on the command line.
"""
import argparse
import csv
import math
import os
import random
import statistics as st
from dataclasses import dataclass, replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- physical process
AMP, SIG_PERIOD, THRESH, KP, SETPOINT = 5.0, 2.0, 2.5, 1.2, 2.5
UMAX = 10.0


def true_signal(t):
    """Sine (amplitude 5, period 2 s) plus a small continuous ripple standing in for sensor noise."""
    return (AMP * math.sin(2 * math.pi * t / SIG_PERIOD)
            + 0.05 * math.sin(2 * math.pi * 3.1 * t) + 0.03 * math.sin(2 * math.pi * 7.7 * t + 1.0))


# ---------------------------------------------------------------- configuration
@dataclass(frozen=True)
class Cfg:
    name: str = "baseline"
    T: float = 0.05                 # nominal sampling period (s) - 20 Hz
    duration: float = 20.0          # simulated run length (s)
    jitter: float = 0.0             # jitter bound J in seconds (uniform(-J, J))
    p_over: float = 0.0             # overrun probability per cycle
    over_lo: float = 0.075          # overrun computation range in seconds (absolute, must exceed T)
    over_hi: float = 0.125
    c_lo: float = 0.002             # normal computation time range (s)
    c_hi: float = 0.006

    @property
    def N(self):
        return int(round(self.duration / self.T))


def simulate(cfg, seed):
    rnd = random.Random(seed)
    N, T = cfg.N, cfg.T
    s, f, c, jit, overrun = [], [], [], [], []
    prev_end = 0.0
    for k in range(N):
        j = rnd.uniform(-cfg.jitter, cfg.jitter) if cfg.jitter > 0 else 0.0
        ck = rnd.uniform(cfg.over_lo, cfg.over_hi) if rnd.random() < cfg.p_over else rnd.uniform(cfg.c_lo, cfg.c_hi)
        sk = max(k * T + j, prev_end, 0.0)
        fk = sk + ck
        s.append(sk); f.append(fk); c.append(ck); jit.append(j); overrun.append(ck > T)
        prev_end = fk
    ideal_t = [k * T for k in range(N)]
    return dict(cfg=cfg, s=s, f=f, c=c, jit=jit, overrun=overrun, ideal_t=ideal_t,
                x_ideal=[true_signal(t) for t in ideal_t], x_real=[true_signal(t) for t in s])


# ---------------------------------------------------------------- controllers
def crossings(times, xs):
    """Indices k where the sample crosses THRESH from below (x[k-1] < THRESH <= x[k])."""
    return [k for k in range(1, len(xs)) if xs[k - 1] < THRESH <= xs[k]]


def true_crossing_times(duration):
    """Exact rising crossings of the continuous signal (fine grid + bisection), for reference."""
    out, dt, t = [], 1e-3, 0.0
    while t < duration:
        a, b = true_signal(t), true_signal(t + dt)
        if a < THRESH <= b:
            lo, hi = t, t + dt
            for _ in range(30):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if true_signal(mid) < THRESH else (lo, mid)
            out.append(hi)
        t += dt
    return out


def match_crossings(ideal_acts, real_acts, window):
    """Greedy one-to-one nearest matching; returns (delays, n_missed)."""
    used, delays, missed = set(), [], 0
    for ti in ideal_acts:
        best, bj = None, None
        for j, tr in enumerate(real_acts):
            if j in used:
                continue
            d = abs(tr - ti)
            if d <= window and (best is None or d < best):
                best, bj = d, j
        if bj is None:
            missed += 1
        else:
            used.add(bj)
            delays.append(real_acts[bj] - ti)
    return delays, missed


def held_rms_diff(t1, u1, t2, u2, duration):
    """Exact RMS of (u1(t) - u2(t)) over [0, duration] for two zero-order-hold signals (u before first update = 0)."""
    pts = sorted(set([0.0, duration] + [t for t in t1 if t < duration] + [t for t in t2 if t < duration]))
    i = j = 0
    a = b = 0.0
    acc = 0.0
    for p, q in zip(pts, pts[1:]):
        while i < len(t1) and t1[i] <= p:
            a = u1[i]; i += 1
        while j < len(t2) and t2[j] <= p:
            b = u2[j]; j += 1
        acc += (a - b) ** 2 * (q - p)
    return math.sqrt(acc / duration)


def rms(t, u, duration):
    return held_rms_diff(t, u, [0.0], [0.0], duration)


def p_out(x):
    return max(-UMAX, min(UMAX, KP * (SETPOINT - x)))


def relay_out(x):
    return 1.0 if x >= THRESH else 0.0


def metrics(sim):
    cfg = sim["cfg"]
    T, s, f, c = cfg.T, sim["s"], sim["f"], sim["c"]
    intervals = [s[k] - s[k - 1] for k in range(1, len(s))]
    over = [max(0.0, ck - T) for ck in c]
    # a cycle "misses the nominal period" when it finishes after the next slot begins
    missed_cyc = sum(1 for k in range(len(s)) if f[k] > (k + 1) * T + 1e-12)
    ic = crossings(sim["ideal_t"], sim["x_ideal"])
    rc = crossings(s, sim["x_real"])
    ideal_acts = [sim["ideal_t"][k] for k in ic]
    real_acts = [f[k] for k in rc]
    delays, missed_x = match_crossings(ideal_acts, real_acts, window=0.5 * SIG_PERIOD)
    truec = true_crossing_times(cfg.duration)
    m = dict(
        mean_int=st.mean(intervals), std_int=st.pstdev(intervals),
        max_over=max(over), pct_missed=100.0 * missed_cyc / len(s),
        pct_gt15=100.0 * sum(1 for d in intervals if d > 1.5 * T) / len(intervals),
        n_cross=len(ic), n_matched=len(delays), n_missed=missed_x,
        mean_delay=(st.mean(delays) if delays else float("nan")),
        mean_abs_delay=(st.mean(abs(d) for d in delays) if delays else float("nan")),
        max_delay=(max(abs(d) for d in delays) if delays else float("nan")),
        n_late_cycle=sum(1 for d in delays if d > T),
        det_vs_true=(st.mean(abs(real_acts[0] - 0) for _ in [0]) if False else None),
    )
    # controller output error (normalised RMS deviation from the ideal controller output)
    for tag, fn in (("relay", relay_out), ("p", p_out)):
        u_i = [fn(x) for x in sim["x_ideal"]]
        u_r = [fn(x) for x in sim["x_real"]]
        ref = rms(sim["ideal_t"], u_i, cfg.duration)
        m["rmsdev_" + tag] = 100.0 * held_rms_diff(sim["ideal_t"], u_i, f, u_r, cfg.duration) / ref
    return m


def average_metrics(cfg, seeds):
    runs = [metrics(simulate(cfg, sd)) for sd in seeds]
    out = {}
    for key in runs[0]:
        if key == "det_vs_true":
            continue
        vals = [r[key] for r in runs if not (isinstance(r[key], float) and math.isnan(r[key]))]
        out[key] = (st.mean(vals) if vals else float("nan"))
    out["max_delay"] = max((r["max_delay"] for r in runs if not math.isnan(r["max_delay"])), default=float("nan"))
    out["max_over"] = max(r["max_over"] for r in runs)
    out["n_cross"] = sum(r["n_cross"] for r in runs)
    out["n_missed"] = sum(r["n_missed"] for r in runs)
    out["n_late_cycle"] = sum(r["n_late_cycle"] for r in runs)
    return out


# ---------------------------------------------------------------- figures
def fig1(sim, path, title, t_zoom=8.0):
    cfg = sim["cfg"]
    T = cfg.T
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2.2, 1]})
    tt = [i * 0.002 for i in range(int(t_zoom / 0.002))]
    a1.plot(tt, [true_signal(t) for t in tt], color="0.6", lw=1.2, label="true signal")
    nz = int(t_zoom / T)
    a1.plot(sim["ideal_t"][:nz], sim["x_ideal"][:nz], "o", mfc="none", color="tab:blue", ms=5, label="ideal samples (k*T)")
    zs = [(t, x) for t, x in zip(sim["s"], sim["x_real"]) if t < t_zoom]
    a1.plot([a for a, _ in zs], [b for _, b in zs], "x", color="tab:red", ms=5, label="actual samples (jitter + overruns)")
    a1.axhline(THRESH, color="tab:green", ls="--", lw=1, label="threshold")
    a1.set_xlim(0, t_zoom); a1.set_ylabel("signal"); a1.legend(loc="upper right", fontsize=8, ncol=2)
    a1.set_title(f"Figure 1 - {title} (first {t_zoom:g} s shown)", fontsize=10); a1.grid(alpha=0.3)
    iv = [sim["s"][k] - sim["s"][k - 1] - T for k in range(1, len(sim["s"]))]
    a2.plot(range(1, len(sim["s"])), iv, color="tab:red", lw=0.8)
    a2.axhline(0, color="black", lw=1)
    a2.set_xlabel("cycle index k"); a2.set_ylabel("interval - T (s)"); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig2(sim, path, title):
    cfg = sim["cfg"]
    ic, rc = crossings(sim["ideal_t"], sim["x_ideal"]), crossings(sim["s"], sim["x_real"])
    ia = [sim["ideal_t"][k] for k in ic]; ra = [sim["f"][k] for k in rc]
    delays, missed = match_crossings(ia, ra, 0.5 * SIG_PERIOD)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.eventplot([ia], lineoffsets=1, linelengths=0.6, colors="tab:blue")
    ax.eventplot([ra], lineoffsets=0, linelengths=0.6, colors="tab:red")
    for t in true_crossing_times(cfg.duration):
        ax.axvline(t, color="0.8", lw=0.8, zorder=0)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["real loop", "ideal loop"]); ax.set_xlim(0, cfg.duration); ax.set_ylim(-0.6, 1.9)
    ax.set_xlabel("time (s); grey lines = true crossings")
    mean_d = st.mean(delays) * 1000 if delays else float("nan")
    max_d = max(abs(d) for d in delays) * 1000 if delays else float("nan")
    ax.set_title(f"Figure 2 - {title}: mean actuation delay {mean_d:.1f} ms, max {max_d:.1f} ms, "
                 f"missed crossings {missed}/{len(ia)}", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- sweep
def fmt_row(label, m, T):
    return (f"{label:<34}{m['mean_int'] * 1000:>8.2f}{m['std_int'] * 1000:>8.2f}{m['max_over'] * 1000:>9.1f}{m['pct_missed']:>8.2f}"
            f"{m['pct_gt15']:>8.2f}{m['mean_delay'] * 1000:>9.2f}{m['mean_abs_delay'] * 1000:>9.2f}{m['max_delay'] * 1000:>9.1f}"
            f"{m['n_late_cycle']:>6}{m['n_missed']:>4}/{m['n_cross']:<4}{m['rmsdev_relay']:>8.1f}{m['rmsdev_p']:>8.1f}")


HEADER = (f"{'configuration':<34}{'mean dt':>8}{'sd dt':>8}{'max ovr':>9}{'%miss':>8}{'%>1.5T':>8}{'mean dly':>9}{'|dly|':>9}"
          f"{'max dly':>9}{'late':>6}{'missed':>9}{'relay%':>8}{'P%':>8}")
UNITS = f"{'':<34}{'(ms)':>8}{'(ms)':>8}{'(ms)':>9}{'cycles':>8}{'cycles':>8}{'(ms)':>9}{'(ms)':>9}{'(ms)':>9}{'>1T':>6}{'crossings':>9}{'rms dev':>8}{'rms dev':>8}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--T", type=float, default=0.05)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--jitter", type=float, default=0.0, help="jitter bound as a fraction of T")
    ap.add_argument("--overrun-prob", type=float, default=0.0)
    ap.add_argument("--overrun-lo", type=float, default=1.5, help="overrun computation time, lower end, in multiples of T")
    ap.add_argument("--overrun-hi", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    if not a.sweep:
        cfg = Cfg("custom", T=a.T, duration=a.duration, jitter=a.jitter * a.T, p_over=a.overrun_prob,
                  over_lo=a.overrun_lo * a.T, over_hi=a.overrun_hi * a.T)
        m = metrics(simulate(cfg, a.seed))
        for k, v in m.items():
            if v is not None:
                print(f"{k:>16}: {v:.4f}" if isinstance(v, float) else f"{k:>16}: {v}")
        return

    T = 0.05
    SEEDS = list(range(1, 51))
    lines = []
    P = lambda s="": (print(s), lines.append(s))
    OV = dict(over_lo=1.5 * T, over_hi=2.5 * T)      # mean overrun computation time 2T
    configs = [
        ("0. sanity: J=0, p=0", Cfg("sanity", jitter=0.0, p_over=0.0, **OV)),
        ("1a. jitter only  J=0.1T", Cfg("j1", jitter=0.1 * T, **OV)),
        ("1b. jitter only  J=0.3T", Cfg("j3", jitter=0.3 * T, **OV)),
        ("1c. jitter only  J=0.5T", Cfg("j5", jitter=0.5 * T, **OV)),
        ("2a. overrun only p=0.025", Cfg("o1", p_over=0.025, **OV)),
        ("2b. overrun only p=0.075", Cfg("o3", p_over=0.075, **OV)),
        ("2c. overrun only p=0.125", Cfg("o5", p_over=0.125, **OV)),
        ("3a. both  J=0.1T, p=0.025", Cfg("b1", jitter=0.1 * T, p_over=0.025, **OV)),
        ("3b. both  J=0.3T, p=0.075", Cfg("b3", jitter=0.3 * T, p_over=0.075, **OV)),
        ("3c. both  J=0.5T, p=0.125", Cfg("b5", jitter=0.5 * T, p_over=0.125, **OV)),
        ("4. stress: p=0.10, overrun 10-40T", Cfg("stress", p_over=0.10, over_lo=10 * T, over_hi=40 * T)),
    ]
    P(f"Mini Project 4.1 sweep: T = {T * 1000:.0f} ms, run length {Cfg().duration:g} s ({Cfg().N} cycles), "
      f"{len(SEEDS)} random seeds per configuration (values are averages over seeds; max = worst seed).")
    P(f"Signal: {AMP:g}*sin(2*pi*t/{SIG_PERIOD:g}) + ripple; threshold {THRESH}; P rule u = {KP}*({SETPOINT} - x), |u| <= {UMAX:g}.")
    P("Normal computation time uniform(2, 6) ms; overrun computation time uniform(1.5T, 2.5T) (mean 2T) except the stress row.")
    P("Equal average disturbance: mean |jitter| = J/2 (0.05T, 0.15T, 0.25T) equals overrun probability x mean overrun time (0.025x2T, 0.075x2T, 0.125x2T).")
    P("'%miss' = cycles finishing after the next slot start; '%>1.5T' = compliance metric (interval > 1.5 T); 'late >1T' = crossings actuated more than one T late;")
    P("'relay%'/'P%' = RMS deviation of the held controller output from the ideal controller's output, as % of the ideal output's own RMS.")
    P()
    P(HEADER); P(UNITS)
    results = {}
    rows = []
    for label, cfg in configs:
        m = average_metrics(cfg, SEEDS)
        results[cfg.name] = m
        P(fmt_row(label, m, T))
        rows.append([label] + [round(m[k], 5) for k in ("mean_int", "std_int", "max_over", "pct_missed", "pct_gt15", "mean_delay",
                                                       "mean_abs_delay", "max_delay", "n_late_cycle", "n_missed", "n_cross", "rmsdev_relay", "rmsdev_p")])
    with open(os.path.join(HERE, "sweep_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["configuration", "mean_interval_s", "std_interval_s", "max_overrun_s", "pct_cycles_missed", "pct_interval_gt_1.5T",
                    "mean_delay_s", "mean_abs_delay_s", "max_delay_s", "crossings_late_gt_1T", "crossings_missed", "crossings_total",
                    "relay_rms_dev_pct", "p_rms_dev_pct"])
        w.writerows(rows)

    # ---- lab task 4: halve T with absolute jitter/overrun magnitudes fixed
    P()
    P("Lab task 4: halve T (50 ms -> 25 ms) holding the ABSOLUTE jitter bound (15 ms) and overrun times (75-125 ms) fixed.")
    P(f"{'configuration':<34}{'J/T':>7}{'mean dt':>9}{'sd dt':>8}{'%>1.5T':>8}{'mean|dly|':>10}{'max dly':>9}{'dly in T':>9}{'missed':>9}{'relay%':>8}{'P%':>8}")
    for Tx in (0.05, 0.025):
        cfg = Cfg(f"halveT_{Tx}", T=Tx, jitter=0.015, p_over=0.05 * 0.05 / Tx * 0 + 0.05, over_lo=0.075, over_hi=0.125)
        m = average_metrics(cfg, SEEDS)
        results[cfg.name] = m
        P(f"{'T = %g ms, J = 15 ms, p = 0.05' % (Tx * 1000):<34}{cfg.jitter / Tx:>7.2f}{m['mean_int'] * 1000:>9.2f}{m['std_int'] * 1000:>8.2f}"
          f"{m['pct_gt15']:>8.2f}{m['mean_abs_delay'] * 1000:>10.2f}{m['max_delay'] * 1000:>9.1f}{m['mean_abs_delay'] / Tx:>9.2f}"
          f"{m['n_missed']:>5}/{m['n_cross']:<4}{m['rmsdev_relay']:>8.1f}{m['rmsdev_p']:>8.1f}")

    # ---- figures for three configurations (seed 3)
    figs = [("jitter_only", "jitter only, J = 0.3T", Cfg("fj", jitter=0.3 * T, **OV)),
            ("overrun_only", "overruns only, p = 0.075", Cfg("fo", p_over=0.075, **OV)),
            ("combined", "jitter J = 0.3T and overruns p = 0.075", Cfg("fb", jitter=0.3 * T, p_over=0.075, **OV)),
            ("stress", "stress: overruns p = 0.10 of 10-40T", Cfg("fs", p_over=0.10, over_lo=10 * T, over_hi=40 * T))]
    P()
    for tag, title, cfg in figs:
        sim = simulate(cfg, 3)
        fig1(sim, os.path.join(HERE, f"fig1_{tag}.png"), title)
        fig2(sim, os.path.join(HERE, f"fig2_{tag}.png"), title)
        P(f"figures written: fig1_{tag}.png, fig2_{tag}.png  ({title})")
    open(os.path.join(HERE, "sweep_output.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
