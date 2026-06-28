from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


KEY_METRICS = [
    "initial_hic_hit_count",
    "edge_supervised_hic_hit_count",
    "edge_supervised_hic_delta_hit_rate",
    "initial_auroc",
    "edge_supervised_auroc",
    "initial_auprc",
    "edge_supervised_auprc",
    "edge_supervised_delta_auprc",
    "initial_hit_count_at_500",
    "edge_supervised_hit_count_at_500",
    "initial_hit_count_at_1000",
    "edge_supervised_hit_count_at_1000",
    "initial_hit_count_at_2000",
    "edge_supervised_hit_count_at_2000",
    "edge_supervised_delta_precision_at_1000",
    "edge_supervised_delta_recall_at_1000",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize edge reranker parameter grid results.")
    parser.add_argument("--grid-root", type=str, required=True, help="Directory containing one subdir per parameter combo.")
    parser.add_argument("--output-tsv", type=str, required=True, help="Grid-level TSV summary.")
    return parser.parse_args()


def _load_params(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "grid_params.json"
    if not path.exists():
        return {"config": run_dir.name}
    with open(path, "r", encoding="utf-8-sig") as f:
        params = json.load(f)
    params.setdefault("config", run_dir.name)
    return params


def _summary_value(summary: pd.DataFrame, metric: str, field: str = "mean") -> float:
    rows = summary.loc[summary["metric"].eq(metric)]
    if rows.empty or field not in rows.columns:
        return float("nan")
    return float(rows.iloc[0][field])


def _summarize_run(run_dir: Path) -> Dict[str, Any] | None:
    summary_path = run_dir / "multi_seed_metrics_summary.tsv"
    if not summary_path.exists():
        return None
    summary = pd.read_csv(summary_path, sep="\t")
    row = _load_params(run_dir)
    row["run_dir"] = str(run_dir)
    for metric in KEY_METRICS:
        row[f"{metric}_mean"] = _summary_value(summary, metric, "mean")
        row[f"{metric}_std"] = _summary_value(summary, metric, "std")

    row["top1000_hit_gain_mean"] = (
        row["edge_supervised_hit_count_at_1000_mean"] - row["initial_hit_count_at_1000_mean"]
    )
    row["top2000_hit_gain_mean"] = (
        row["edge_supervised_hit_count_at_2000_mean"] - row["initial_hit_count_at_2000_mean"]
    )
    row["auroc_delta_mean"] = row["edge_supervised_auroc_mean"] - row["initial_auroc_mean"]
    row["auprc_delta_mean"] = row["edge_supervised_auprc_mean"] - row["initial_auprc_mean"]
    return row


def main() -> None:
    args = parse_args()
    grid_root = Path(args.grid_root)
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(path for path in grid_root.iterdir() if path.is_dir()):
        row = _summarize_run(run_dir)
        if row is not None:
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No multi_seed_metrics_summary.tsv files found under {grid_root}")

    df = pd.DataFrame(rows)
    sort_cols = [col for col in ["top1000_hit_gain_mean", "auprc_delta_mean"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False, False]).reset_index(drop=True)

    output_path = Path(args.output_tsv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved grid summary to {output_path}")

    show_cols = [
        "config",
        "delta_logit_scale",
        "delta_l2_weight",
        "ranking_loss_weight",
        "negative_sampling",
        "initial_hit_count_at_1000_mean",
        "edge_supervised_hit_count_at_1000_mean",
        "top1000_hit_gain_mean",
        "initial_auprc_mean",
        "edge_supervised_auprc_mean",
        "auprc_delta_mean",
        "auroc_delta_mean",
    ]
    present = [col for col in show_cols if col in df.columns]
    print(df[present].to_string(index=False))


if __name__ == "__main__":
    main()
