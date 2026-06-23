from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize multi-seed edge-supervised evaluation JSON files.")
    parser.add_argument("--metrics-jsons", nargs="*", default=[], help="Explicit metrics JSON paths.")
    parser.add_argument("--metrics-glob", type=str, default="", help="Glob pattern for metrics JSON files.")
    parser.add_argument("--output-tsv", type=str, required=True, help="Per-seed TSV output path.")
    parser.add_argument("--summary-tsv", type=str, default="", help="Mean/std TSV output path.")
    return parser.parse_args()


def _collect_paths(explicit_paths: Iterable[str], pattern: str) -> List[Path]:
    paths = [Path(path) for path in explicit_paths]
    if pattern:
        paths.extend(Path(path) for path in glob.glob(pattern))
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise FileNotFoundError("No metrics JSON files were provided or matched.")
    return unique


def _safe_get(obj: Dict[str, Any], path: str, default: Any = math.nan) -> Any:
    cur: Any = obj
    for key in path.split("."):
        if cur is None or not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _seed_from_path(path: Path) -> str:
    text = str(path)
    matches = re.findall(r"(?:seed|s)(\d+)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1]
    return path.parent.name


def _flatten_metrics(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    row: Dict[str, Any] = {
        "seed": _seed_from_path(path),
        "metrics_json": str(path),
        "topk": data.get("topk", math.nan),
        "hic_split_name": data.get("hic_split_name", ""),
        "hic_loops": data.get("hic_loops_after_chrom_filter", math.nan),
        "edge_label_col": data.get("edge_label_col", ""),
    }

    for graph in ["initial", "optimized", "edge_supervised"]:
        row[f"{graph}_hic_hit_count"] = _safe_get(data, f"hic_overlap_metrics.{graph}.hit_count")
        row[f"{graph}_hic_hit_rate"] = _safe_get(data, f"hic_overlap_metrics.{graph}.hit_rate")
        row[f"{graph}_auroc"] = _safe_get(data, f"edge_label_ranking_metrics.{graph}.auroc")
        row[f"{graph}_auprc"] = _safe_get(data, f"edge_label_ranking_metrics.{graph}.auprc")

    row["edge_supervised_hic_delta_hit_rate"] = row["edge_supervised_hic_hit_rate"] - row["initial_hic_hit_rate"]
    row["edge_supervised_delta_auprc"] = row["edge_supervised_auprc"] - row["initial_auprc"]

    topks = data.get("edge_metric_topks", [])
    for k in topks:
        for graph in ["initial", "optimized", "edge_supervised"]:
            prefix = f"edge_label_ranking_metrics.{graph}"
            row[f"{graph}_hit_count_at_{k}"] = _safe_get(data, f"{prefix}.hit_count_at_{k}")
            row[f"{graph}_precision_at_{k}"] = _safe_get(data, f"{prefix}.precision_at_{k}")
            row[f"{graph}_recall_at_{k}"] = _safe_get(data, f"{prefix}.recall_at_{k}")
            row[f"{graph}_enrichment_at_{k}"] = _safe_get(data, f"{prefix}.enrichment_at_{k}")
        row[f"edge_supervised_delta_precision_at_{k}"] = (
            row[f"edge_supervised_precision_at_{k}"] - row[f"initial_precision_at_{k}"]
        )
        row[f"edge_supervised_delta_recall_at_{k}"] = row[f"edge_supervised_recall_at_{k}"] - row[f"initial_recall_at_{k}"]
    return row


def _write_summary(df: pd.DataFrame, output_path: Path) -> None:
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != "seed"]
    rows = []
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": col,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "n": int(len(values)),
            }
        )
    summary = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, sep="\t", index=False)


def main() -> None:
    args = parse_args()
    paths = _collect_paths(args.metrics_jsons, args.metrics_glob)
    rows = [_flatten_metrics(path) for path in paths]
    df = pd.DataFrame(rows).sort_values("seed")

    output_path = Path(args.output_tsv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved per-seed metrics to {output_path}")

    summary_path = Path(args.summary_tsv) if args.summary_tsv else output_path.with_name(output_path.stem + "_summary.tsv")
    _write_summary(df, summary_path)
    print(f"Saved mean/std summary to {summary_path}")

    key_cols = [
        "seed",
        "initial_hic_hit_count",
        "edge_supervised_hic_hit_count",
        "initial_auprc",
        "edge_supervised_auprc",
        "edge_supervised_delta_auprc",
    ]
    present = [col for col in key_cols if col in df.columns]
    print(df[present].to_string(index=False))


if __name__ == "__main__":
    main()
