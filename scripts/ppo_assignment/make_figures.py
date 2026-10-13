#!/usr/bin/env python3
"""Build the report figures from the CSVs written by run_one.py."""
import csv, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "runs" / "ppo_assignment"
FIG = DATA / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.size": 7, "axes.grid": True, "grid.alpha": 0.3,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1, "legend.frameon": False, "figure.dpi": 200,
})
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def load(tag):
    with (DATA / f"{tag}.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for k in rows[0]:
        out[k] = np.array([float(r[k]) if r.get(k) not in (None, "", "nan") else np.nan for r in rows])
    return out


def smooth(y, w):
    if w <= 1:
        return y
    k = np.ones(w) / w
    return np.convolve(np.concatenate([np.full(w - 1, y[0]), y]), k, mode="valid")


def series(tags, column, w, xcol="global_step"):
    runs = [load(t) for t in tags]
    n = min(len(r[xcol]) for r in runs)
    x = runs[0][xcol][:n]
    ys = []
    for r in runs:
        y = r[column][:n]
        m = np.isnan(y)
        if m.any() and (~m).any():
            y = np.where(m, np.interp(x, x[~m], y[~m]), y)
        ys.append(smooth(np.nan_to_num(y), w))
    ys = np.stack(ys)
    return x, ys.mean(0), ys.min(0), ys.max(0)


def panel(ax, groups, column, w, title, ylabel, xcol="global_step", xdiv=1e3,
          xlabel="passos (mil)", hline=None):
    for i, (label, tags) in enumerate(groups):
        x, mean, lo, hi = series(tags, column, w, xcol)
        c = COLORS[i % len(COLORS)]
        ax.plot(x / xdiv, mean, color=c, label=label)
        ax.fill_between(x / xdiv, lo, hi, color=c, alpha=0.15, linewidth=0)
    if hline is not None:
        ax.axhline(hline, color="0.35", linestyle="--", linewidth=0.7)
    ax.set_title(title, fontsize=7.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def figure(name, groups, panels, width=7.0, height=1.95, legend_loc="best", legend_on=0):
    fig, axes = plt.subplots(1, len(panels), figsize=(width, height))
    axes = np.atleast_1d(axes)
    for ax, spec in zip(axes, panels):
        panel(ax, groups, **spec)
    axes[legend_on].legend(fontsize=5.8, loc=legend_loc)
    fig.tight_layout(pad=0.4)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / (name + '.png')}")


# ---------------------------------------------------------------------------
# Report figures. BEST is filled in once the candidate runs have finished.
# ---------------------------------------------------------------------------

BASE = ["ppo_base_s1", "ppo_base_s2"]
BEST = None   # set by main() from the winning candidate tag prefix
A2C = ["a2_s1", "a2_s2"]


def steps_to_reach(tags, threshold):
    """Mean environment steps needed to first reach a return threshold."""
    out = []
    for t in tags:
        d = load(t)
        r = np.nan_to_num(d["charts/episodic_return_mean_last100"], nan=-1e9)
        idx = np.where(r >= threshold)[0]
        out.append(d["global_step"][idx[0]] if len(idx) else np.nan)
    return np.nanmean(out) if not all(np.isnan(out)) else np.nan


def main(best_prefix):
    best = [f"{best_prefix}_s1", f"{best_prefix}_s2"]

    # Part 4, item 4: the return curves of the starting and the final config
    figure("p4_return", [("config inicial", BASE), ("config final", best)], [
        dict(column="charts/episodic_return_mean_last100", w=15,
             title="charts/episodic_return_mean_last100", ylabel="retorno", hline=200.0),
        dict(column="losses/explained_variance", w=25,
             title="losses/explained_variance", ylabel="variância explicada"),
        dict(column="losses/entropy", w=25, title="losses/entropy", ylabel="entropia"),
    ], legend_loc="upper left")

    # Part 4, item 2: the diagnostics that motivated the changes
    figure("p4_diag", [("config inicial", BASE),
                       ("+ lr 1e-3", ["p_lr1e3_s1", "p_lr1e3_s2"]),
                       ("config final", best)], [
        dict(column="losses/clipfrac", w=25, title="losses/clipfrac", ylabel="clipfrac"),
        dict(column="losses/approx_kl", w=25, title="losses/approx_kl", ylabel="KL aprox."),
        dict(column="losses/grad_norm", w=25, title="losses/grad_norm", ylabel="norma do grad."),
    ], legend_loc="upper right")

    # Part 5: PPO vs A2C, on environment steps and on wall-clock time
    figure("p5_compare", [("PPO (final)", best), ("A2C", A2C)], [
        dict(column="charts/episodic_return_mean_last100", w=15,
             title="retorno × passos de ambiente", ylabel="retorno", hline=200.0),
        dict(column="charts/episodic_return_mean_last100", w=15, xcol="_runtime", xdiv=60.0,
             xlabel="tempo (min)", title="retorno × tempo de relógio", ylabel="retorno", hline=200.0),
        dict(column="charts/SPS", w=25, title="charts/SPS", ylabel="passos de ambiente / s"),
    ], legend_loc="upper left")

    for label, tags in [("PPO final", best), ("A2C", A2C)]:
        marks = " ".join(f"{th:>4}: {steps_to_reach(tags, th)/1000:7.0f}k" for th in (0, 100, 200))
        print(f"{label:<12} passos até retorno  {marks}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "c1")
