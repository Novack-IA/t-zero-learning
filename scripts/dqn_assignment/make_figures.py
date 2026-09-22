#!/usr/bin/env python3
"""Build the report figures from the CSVs written by run_one.py.

One figure per question; each curve is the mean over the two seeds of a sweep
value, with a band covering the seed-to-seed spread.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "runs" / "dqn_assignment"
FIG_DIR = REPO_ROOT / "runs" / "dqn_assignment" / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.1,
    "legend.frameon": False,
    "figure.dpi": 200,
})

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def load(tag):
    path = DATA_DIR / f"{tag}.csv"
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for key in rows[0]:
        vals = [float(r[key]) if r[key] != "" else np.nan for r in rows]
        out[key] = np.array(vals)
    return out


def eval_metrics(tag):
    path = DATA_DIR / f"{tag}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def smooth(y, w):
    if w <= 1:
        return y
    kernel = np.ones(w) / w
    pad = np.concatenate([np.full(w - 1, y[0]), y])
    return np.convolve(pad, kernel, mode="valid")


def series(tags, column, w):
    """Mean / min / max over seeds, on the shortest common step grid."""
    runs = [load(t) for t in tags]
    n = min(len(r["global_step"]) for r in runs)
    x = runs[0]["global_step"][:n]
    ys = []
    for r in runs:
        y = r[column][:n]
        # the return series is empty until the first episode finishes
        mask = np.isnan(y)
        if mask.any():
            y = np.where(mask, np.interp(x, x[~mask], y[~mask]) if (~mask).any() else 0.0, y)
        ys.append(smooth(y, w))
    ys = np.stack(ys)
    return x, ys.mean(0), ys.min(0), ys.max(0)


def panel(ax, groups, column, w, title, ylabel, logy=False, hline=None):
    for i, (label, tags) in enumerate(groups):
        x, mean, lo, hi = series(tags, column, w)
        c = COLORS[i % len(COLORS)]
        ax.plot(x / 1000.0, mean, color=c, label=label)
        ax.fill_between(x / 1000.0, lo, hi, color=c, alpha=0.15, linewidth=0)
    if hline is not None:
        # true value bound in CartPole: r=1 per step, sum_t gamma^t = 1/(1-gamma)
        ax.axhline(hline, color="0.35", linestyle="--", linewidth=0.7)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=7.5)
    ax.set_xlabel("passos (mil)")
    ax.set_ylabel(ylabel)


def figure(name, groups, panels, width=7.0, height=1.9, legend_loc="best"):
    fig, axes = plt.subplots(1, len(panels), figsize=(width, height))
    for ax, spec in zip(np.atleast_1d(axes), panels):
        panel(ax, groups, *spec)
    np.atleast_1d(axes)[0].legend(fontsize=6.0, loc=legend_loc)
    fig.tight_layout(pad=0.4)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def table(groups):
    print(f"\n{'configuração':<28}{'retorno final (últimos 100)':>28}{'eval greedy (10 ep)':>26}")
    for label, tags in groups:
        finals, evals = [], []
        for t in tags:
            d = load(t)
            y = d["episodic_return_mean_last100"]
            y = y[~np.isnan(y)]
            # average the last 20 logged points: the raw last value is noisy
            finals.append(y[-20:].mean())
            m = eval_metrics(t)
            if m:
                evals.append((m["eval/mean_return"], m["eval/std_return"]))
        ev = ", ".join(f"{a:.0f}+-{b:.0f}" for a, b in evals) or "-"
        print(f"{label:<28}{np.mean(finals):>28.0f}{ev:>26}")


def main():
    base = ["base_s1", "base_s2"]

    q1 = [
        ("freq = 1", ["q1_tnf1_s1", "q1_tnf1_s2"]),
        ("freq = 100", ["q1_tnf100_s1", "q1_tnf100_s2"]),
        ("freq = 500 (baseline)", base),
        ("freq = 5000", ["q1_tnf5000_s1", "q1_tnf5000_s2"]),
    ]
    figure("q1", q1, [
        ("episodic_return_mean_last100", 5, "charts/episodic_return_mean_last100", "retorno"),
        ("td_loss", 40, "losses/td_loss (suavizado)", "perda TD", True),
        ("q_values", 40, "losses/q_values", "Q médio", False, 100.0),
    ])
    table(q1)

    q2 = [
        ("buffer = 32", ["q2_buf32_s1", "q2_buf32_s2"]),
        ("buffer = 128", ["q2_buf128_s1", "q2_buf128_s2"]),
        ("buffer = 1.000", ["q2_buf1000_s1", "q2_buf1000_s2"]),
        ("buffer = 10.000 (baseline)", base),
        ("buffer = 200.000", ["q2_buf200000_s1", "q2_buf200000_s2"]),
    ]
    figure("q2", q2, [
        ("episodic_return_mean_last100", 5, "charts/episodic_return_mean_last100", "retorno"),
        ("q_values", 40, "losses/q_values", "Q médio", False, 100.0),
    ], legend_loc="upper left")
    table(q2)

    q3 = [
        ("fração = 0,02", ["q3_ef002_s1", "q3_ef002_s2"]),
        ("fração = 0,2", ["q3_ef020_s1", "q3_ef020_s2"]),
        ("fração = 0,5 (baseline)", base),
        ("fração = 0,9", ["q3_ef090_s1", "q3_ef090_s2"]),
    ]
    figure("q3", q3, [
        ("episodic_return_mean_last100", 5, "charts/episodic_return_mean_last100", "retorno"),
        ("epsilon", 1, "charts/epsilon", "epsilon"),
        ("q_values", 40, "losses/q_values", "Q médio", False, 100.0),
    ])
    table(q3)


if __name__ == "__main__":
    main()
