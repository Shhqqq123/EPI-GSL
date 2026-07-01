from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize multi-chromosome LOCO metrics.")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--output-tsv", type=str, required=True)
    return parser.parse_args()


def _get_metric(metrics: Dict[str, Any], group: str, name: str, default: float = float("nan")) -> float:
    value = metrics.get(group)
    if not isinstance(value, dict):
        return default
    value = value.get(name, default)
    return float(value) if value is not None else default


def _ranking_metrics(metrics: Dict[str, Any], source: str) -> Dict[str, Any]:
    ranking = metrics.get("edge_label_ranking_metrics", {})
    source_metrics = ranking.get(source, {}) if isinstance(ranking, dict) else {}
    if not isinstance(source_metrics, dict):
        source_metrics = {}
    return source_metrics


def _hit_at(source_metrics: Dict[str, Any], k: int) -> float:
    return float(source_metrics.get(f"hit_count_at_{k}", float("nan")))


def _precision_at(source_metrics: Dict[str, Any], k: int) -> float:
    return float(source_metrics.get(f"precision_at_{k}", float("nan")))


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _summarize_metrics(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    train_dir = path.parent.parent / "train"
    run_config = _load_json_if_exists(train_dir / "run_config.json")
    best_validation = _load_json_if_exists(train_dir / "best_validation.json")
    score_blend = _load_json_if_exists(train_dir / "score_blend_selection.json")

    initial_overlap = metrics.get("hic_overlap_metrics", {}).get("initial", {})
    learned_overlap = metrics.get("hic_overlap_metrics", {}).get("edge_supervised", {})
    initial_rank = _ranking_metrics(metrics, "initial")
    learned_rank = _ranking_metrics(metrics, "edge_supervised")

    chrom = str(metrics.get("hic_split_name", path.parent.name))
    row: Dict[str, Any] = {
        "chrom": chrom,
        "metrics_json": str(path),
        "edge_score_col": metrics.get("edge_score_col", "final_score"),
        "validation_chroms": run_config.get("validation_chroms", ""),
        "best_validation_epoch": best_validation.get("best_epoch", float("nan")),
        "score_blend_alpha": score_blend.get("best_alpha", run_config.get("selected_score_blend_alpha", float("nan"))),
        "hic_loops_after_chrom_filter": metrics.get("hic_loops_after_chrom_filter", float("nan")),
        "num_edges": initial_rank.get("num_edges", float("nan")),
        "positive_edges": initial_rank.get("positive_edges", float("nan")),
        "initial_top1000_hits": float(initial_overlap.get("hit_count", float("nan"))),
        "learned_top1000_hits": float(learned_overlap.get("hit_count", float("nan"))),
        "initial_auroc": float(initial_rank.get("auroc", float("nan"))),
        "learned_auroc": float(learned_rank.get("auroc", float("nan"))),
        "initial_auprc": float(initial_rank.get("auprc", float("nan"))),
        "learned_auprc": float(learned_rank.get("auprc", float("nan"))),
    }

    for k in [500, 1000, 2000]:
        row[f"initial_hit_at_{k}"] = _hit_at(initial_rank, k)
        row[f"learned_hit_at_{k}"] = _hit_at(learned_rank, k)
        row[f"hit_gain_at_{k}"] = row[f"learned_hit_at_{k}"] - row[f"initial_hit_at_{k}"]
        row[f"initial_precision_at_{k}"] = _precision_at(initial_rank, k)
        row[f"learned_precision_at_{k}"] = _precision_at(learned_rank, k)

    row["top1000_hit_gain"] = row["learned_top1000_hits"] - row["initial_top1000_hits"]
    row["auroc_delta"] = row["learned_auroc"] - row["initial_auroc"]
    row["auprc_delta"] = row["learned_auprc"] - row["initial_auprc"]
    return row


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    metric_paths = sorted(output_root.glob("heldout_*/pred_*/eval_metrics.json"))
    if not metric_paths:
        raise FileNotFoundError(f"No heldout_*/pred_*/eval_metrics.json files found under {output_root}")

    rows: List[Dict[str, Any]] = [_summarize_metrics(path) for path in metric_paths]
    df = pd.DataFrame(rows)
    output_path = Path(args.output_tsv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved LOCO summary to {output_path}")

    show_cols = [
        "chrom",
        "validation_chroms",
        "best_validation_epoch",
        "score_blend_alpha",
        "initial_top1000_hits",
        "learned_top1000_hits",
        "top1000_hit_gain",
        "initial_hit_at_2000",
        "learned_hit_at_2000",
        "hit_gain_at_2000",
        "initial_auprc",
        "learned_auprc",
        "auprc_delta",
        "initial_auroc",
        "learned_auroc",
        "auroc_delta",
    ]
    print(df[show_cols].to_string(index=False))
    means = df[["top1000_hit_gain", "hit_gain_at_2000", "auprc_delta", "auroc_delta"]].mean(numeric_only=True)
    print("\nMean deltas:")
    print(means.to_string())


if __name__ == "__main__":
    main()
