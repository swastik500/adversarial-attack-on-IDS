import argparse
import glob
import json
import math
import os
from typing import Dict, List

import numpy as np


METRIC_KEYS = [
    "accuracy_before",
    "accuracy_after",
    "fnr_before",
    "fnr_after",
    "fooling_rate",
]


def load_run_file(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Supports both direct metric files and nested run records.
    if "metrics" in obj and isinstance(obj["metrics"], dict):
        return obj["metrics"]
    return obj


def summarize(values: List[float]) -> Dict:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci95_half = float(1.96 * std / math.sqrt(n)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": mean - ci95_half,
        "ci95_high": mean + ci95_half,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate repeated run metrics and report mean/std/95% CI.")
    parser.add_argument("--runs-dir", default="trained_models/forward_runs")
    parser.add_argument("--pattern", default="metrics_*.json")
    parser.add_argument("--output", default="trained_models/forward_significance.json")
    parser.add_argument("--min-runs", type=int, default=3)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.runs_dir, args.pattern)))
    if len(paths) < args.min_runs:
        raise RuntimeError(
            f"Need at least {args.min_runs} run files in '{args.runs_dir}' (found {len(paths)})."
        )

    collected = {k: [] for k in METRIC_KEYS}
    for path in paths:
        run_metrics = load_run_file(path)
        for key in METRIC_KEYS:
            if key not in run_metrics:
                continue
            collected[key].append(float(run_metrics[key]))

    summary = {
        "runs_dir": args.runs_dir,
        "pattern": args.pattern,
        "n_files": len(paths),
        "metrics": {},
    }

    for key, vals in collected.items():
        if not vals:
            continue
        summary["metrics"][key] = summarize(vals)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Statistical significance summary")
    print(f"Runs analyzed: {summary['n_files']}")
    for key in METRIC_KEYS:
        if key not in summary["metrics"]:
            continue
        row = summary["metrics"][key]
        print(
            f"- {key}: mean={row['mean']:.6f}, std={row['std']:.6f}, "
            f"95% CI=[{row['ci95_low']:.6f}, {row['ci95_high']:.6f}]"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
