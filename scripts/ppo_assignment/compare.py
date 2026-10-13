#!/usr/bin/env python3
"""One-line-per-run comparison table of the key end-of-training diagnostics."""
import csv, json, sys
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[2] / "runs" / "ppo_assignment"


def load(tag):
    with (DATA / f"{tag}.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for k in rows[0]:
        out[k] = np.array([float(r[k]) if r.get(k) not in (None, "", "nan") else np.nan for r in rows])
    return out


def row(tag):
    d = load(tag)
    j = json.loads((DATA / f"{tag}.json").read_text())
    ret = d["charts/episodic_return_mean_last100"]
    ev = d.get("eval/mean_return", np.array([np.nan]))
    ev = ev[~np.isnan(ev)]
    # diagnostics are read mid-training: anneal_lr drives the learning rate
    # to 0 at the end, so kl and clipfrac always collapse there
    n = len(ret); lo, hi = int(0.2 * n), int(0.6 * n)
    g = lambda k: np.nanmean(d[k][lo:hi]) if k in d else np.nan
    return dict(
        tag=tag, ret_fim=np.nanmean(ret[-20:]), ret_max=np.nanmax(ret),
        eval=ev[-1] if len(ev) else np.nan,
        kl=g("losses/approx_kl"), clipfrac=g("losses/clipfrac"),
        entropy=g("losses/entropy"), ev_var=g("losses/explained_variance"),
        gnorm=g("losses/grad_norm"), seg=j.get("wall_clock_seconds", np.nan),
    )


def main(tags):
    hdr = f"{'run':<22}{'ret_fim':>9}{'ret_max':>9}{'eval':>9}{'kl':>9}{'clipfr':>8}{'entrop':>8}{'expl_var':>9}{'gnorm':>8}{'seg':>6}"
    print(hdr); print("-" * len(hdr))
    for t in tags:
        r = row(t)
        print(f"{r['tag']:<22}{r['ret_fim']:>9.1f}{r['ret_max']:>9.1f}{r['eval']:>9.1f}"
              f"{r['kl']:>9.4f}{r['clipfrac']:>8.3f}{r['entropy']:>8.3f}{r['ev_var']:>9.3f}"
              f"{r['gnorm']:>8.2f}{r['seg']:>6.0f}")


if __name__ == "__main__":
    main(sys.argv[1:])
