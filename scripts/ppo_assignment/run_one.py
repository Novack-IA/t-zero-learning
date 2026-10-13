#!/usr/bin/env python3
"""Run one configuration of any algorithm and log its metrics to a local CSV.

The harness logs through Weights & Biases. Instead of touching the algorithm
files, this installs a stub ``wandb`` module before the algorithm imports it,
so every metric the harness would have sent to W&B (including the final
eval/* numbers) is captured on disk with its wall-clock time.

    python scripts/ppo_assignment/run_one.py --config ppo_lunarlander \
        --tag ppo_base_s1 --override seed=1
"""
import argparse
import csv
import json
import sys
import time
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "runs" / "ppo_assignment"


def install_wandb_stub(rows: list, summary: dict):
    """Replace wandb with a sink that records every logged metric dict."""
    start = time.time()
    stub = types.ModuleType("wandb")

    def init(**kwargs):
        summary["config"] = {k: v for k, v in kwargs.get("config", {}).items()
                             if not isinstance(v, (dict, list))}
        return stub

    def log(metrics, step=None, **kwargs):
        row = dict(metrics)
        row["_step"] = step
        # wall-clock seconds since process start: the report compares PPO and
        # A2C on time as well as on environment steps
        row["_runtime"] = time.time() - start
        rows.append(row)

    stub.init = init
    stub.log = log
    stub.finish = lambda *a, **k: None
    stub.run = None
    sys.modules["wandb"] = stub


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tag", required=True, help="output basename under runs/ppo_assignment")
    parser.add_argument("--override", nargs="*", default=[])
    cli = parser.parse_args()

    rows: list = []
    summary: dict = {}
    install_wandb_stub(rows, summary)

    from core.config_loader import apply_overrides, get_config_path, load_config, _import_algorithm

    args, algo_name = load_config(get_config_path(cli.config))
    args.exp_name = f"ppo_assignment/{cli.tag}"
    # keep track=true so the harness feeds the stub; videos are not in the report
    apply_overrides(args, ["track=true", "capture_video=false"] + list(cli.override))

    _, algo_main = _import_algorithm(algo_name)
    t0 = time.time()
    try:
        algo_main(args)
    finally:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fields, seen = [], set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    fields.append(k)
        with (OUT_DIR / f"{cli.tag}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        summary["wall_clock_seconds"] = time.time() - t0
        summary["tag"] = cli.tag
        (OUT_DIR / f"{cli.tag}.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[{cli.tag}] done in {summary['wall_clock_seconds']:.0f}s")


if __name__ == "__main__":
    main()
