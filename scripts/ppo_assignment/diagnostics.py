#!/usr/bin/env python3
"""Print the end-of-training diagnostics of one or more runs."""
import csv, json, sys
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[2] / "runs" / "ppo_assignment"

COLS = ["charts/episodic_return_mean_last100", "losses/approx_kl", "losses/clipfrac",
        "losses/entropy", "losses/explained_variance", "losses/value_loss",
        "charts/advantage_std", "losses/grad_norm"]


def load(tag):
    with (DATA / f"{tag}.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for k in rows[0]:
        out[k] = np.array([float(r[k]) if r.get(k) not in (None, "", "nan") else np.nan for r in rows])
    return out


def summary(tag):
    j = json.loads((DATA / f"{tag}.json").read_text())
    return j.get("wall_clock_seconds"), j.get("eval", {})


def main(tags):
    for tag in tags:
        d = load(tag)
        n = len(d["global_step"])
        lo, hi = int(n * 0.25), int(n * 0.75)
        print(f"\n=== {tag} ===  pontos={n}  passos={d['global_step'][-1]:.0f}")
        wc, _ = summary(tag)
        ret = d["charts/episodic_return_mean_last100"]
        print(f"  retorno: inicio={np.nanmean(ret[:5]):8.1f}  meio={np.nanmean(ret[lo:hi]):8.1f}  fim={np.nanmean(ret[-10:]):8.1f}  max={np.nanmax(ret):8.1f}")
        print(f"  tempo total: {wc:.0f}s")
        for c in COLS[1:]:
            if c not in d:
                continue
            v = d[c]
            print(f"  {c:35s} inicio={np.nanmean(v[:10]):10.4f}  meio={np.nanmean(v[lo:hi]):10.4f}  fim={np.nanmean(v[-10:]):10.4f}")
        # eval metrics land in the last rows
        for key in ("eval/mean_return", "eval/std_return"):
            if key in d:
                vals = d[key][~np.isnan(d[key])]
                if len(vals):
                    print(f"  {key:35s} {vals[-1]:.1f}")


if __name__ == "__main__":
    main(sys.argv[1:])
