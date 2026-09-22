#!/usr/bin/env python3
"""Run one DQN configuration and log its metrics to a local CSV.

The assignment harness logs to Weights & Biases; this wrapper reproduces the
same six metric series on disk so the sweep figures can be built offline.
Nothing in ``algorithms/dqn.py`` is touched: the DQN class is subclassed and
only its logging hooks are extended.

    python scripts/dqn_assignment/run_one.py --tag q1_tnf500_s1 \
        --override dqn.target_network_frequency=500 seed=1
"""
import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from algorithms.dqn import DQN  # noqa: E402
from core.config_loader import apply_overrides, get_config_path, load_config  # noqa: E402

OUT_DIR = REPO_ROOT / "runs" / "dqn_assignment"

FIELDS = [
    "global_step",
    "episodic_return_mean_last100",
    "episodic_length_mean_last100",
    "td_loss",
    "q_values",
    "epsilon",
]


class CsvLoggingDQN(DQN):
    """DQN plus a CSV sink on the existing ``_log_metrics`` hook."""

    def __init__(self, args, csv_path: Path):
        super().__init__(args)
        self._csv_path = csv_path
        self._rows: list[dict] = []
        self._eval_metrics: dict = {}

    def _log_metrics(self, pbar, *, epsilon, loss, old_val) -> None:
        super()._log_metrics(pbar, epsilon=epsilon, loss=loss, old_val=old_val)
        import numpy as np

        row = {
            "global_step": self.global_step,
            "td_loss": loss.item(),
            "q_values": old_val.mean().item(),
            "epsilon": float(epsilon),
            "episodic_return_mean_last100": (
                float(np.mean(self.recent_ep_returns)) if self.recent_ep_returns else ""
            ),
            "episodic_length_mean_last100": (
                float(np.mean(self.recent_ep_lengths)) if self.recent_ep_lengths else ""
            ),
        }
        self._rows.append(row)

    def evaluate(self, *a, **kw):
        results = super().evaluate(*a, **kw)
        self._eval_metrics = dict(results.get("metrics") or {})
        return results

    def flush(self):
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self._csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self._rows)
        meta = self._csv_path.with_suffix(".json")
        meta.write_text(json.dumps(self._eval_metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dqn_cartpole")
    parser.add_argument("--tag", required=True, help="output basename under runs/dqn_assignment")
    parser.add_argument("--override", nargs="*", default=[])
    cli = parser.parse_args()

    args, _ = load_config(get_config_path(cli.config))
    args.exp_name = f"dqn_assignment/{cli.tag}"
    # metrics go to CSV, not W&B; videos are not part of the report
    apply_overrides(args, ["track=false", "capture_video=false"] + list(cli.override))

    run = CsvLoggingDQN(args, OUT_DIR / f"{cli.tag}.csv")
    run.initialize()
    try:
        run.train()
    finally:
        run.flush()
    print(f"[{cli.tag}] done -> {OUT_DIR / (cli.tag + '.csv')}")


if __name__ == "__main__":
    main()
