from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_utils import build_candidate_adj_from_abc_edges, load_abc_edges, load_peak_node_tables
from model import EdgeResidualReranker, PeakLevelIDGLPyG
from train import _edge_feature_frame, _parse_edge_feature_cols, build_edge_score_adj, build_edge_score_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a trained edge-supervised model to a held-out chromosome.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--train-bundle", type=str, required=True, help="Training ep_idgl_outputs.pt containing feature columns.")
    parser.add_argument("--run-config", type=str, required=True, help="Training run_config.json.")
    parser.add_argument("--promoter-path", type=str, default=str(CURRENT_DIR.parent / "promoter_nodes_full.tsv"))
    parser.add_argument("--re-path", type=str, default=str(CURRENT_DIR.parent / "re_nodes_full.tsv"))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--abc-edges", type=str, required=True, help="ABC edge table for the held-out chromosome.")
    parser.add_argument("--edge-table", type=str, default="", help="Edge table used for edge-supervised scores. Defaults to --abc-edges.")
    parser.add_argument("--abc-score-col", type=str, default="abc_score")
    parser.add_argument("--chrom", type=str, required=True)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-dense-output-nodes", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _standardize_with_stats(values: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, -1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, -1)
    std_arr = np.where(std_arr <= 1e-6, 1.0, std_arr)
    return ((values - mean_arr) / std_arr).astype(np.float32)


def _load_edges_for_scoring(
    edge_table_path: str,
    node_table: pd.DataFrame,
    edge_feature_cols: List[str],
    edge_feature_mean: List[float],
    edge_feature_std: List[float],
    abc_score_col: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    edge_df = pd.read_csv(edge_table_path, sep="\t").copy()
    required = ["re_node_id", "promoter_node_id", abc_score_col]
    missing = [col for col in required if col not in edge_df.columns]
    if missing:
        raise KeyError(f"Edge table is missing required columns: {missing}")

    node_to_idx = {str(node_id): idx for idx, node_id in enumerate(node_table["node_id"].astype(str))}
    src_idx = edge_df["re_node_id"].astype(str).map(node_to_idx)
    dst_idx = edge_df["promoter_node_id"].astype(str).map(node_to_idx)
    keep = src_idx.notna() & dst_idx.notna()
    edge_df = edge_df.loc[keep].reset_index(drop=True)
    src = src_idx.loc[keep].astype(int).to_numpy(dtype=np.int64)
    dst = dst_idx.loc[keep].astype(int).to_numpy(dtype=np.int64)

    edge_features = _edge_feature_frame(edge_df, edge_feature_cols).to_numpy(dtype=np.float32)
    edge_features = _standardize_with_stats(edge_features, edge_feature_mean, edge_feature_std)
    base_scores = pd.to_numeric(edge_df[abc_score_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    edge_index = torch.from_numpy(np.vstack([src, dst])).long()
    edge_attr = torch.from_numpy(edge_features).float()
    edge_base_score = torch.from_numpy(base_scores).float()
    return edge_index, edge_attr, edge_base_score, edge_df


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_bundle: Dict = torch.load(args.train_bundle, map_location="cpu")
    with open(args.run_config, "r", encoding="utf-8") as f:
        train_config = json.load(f)

    feature_cols = train_bundle["feature_cols"]
    edge_feature_cols = train_bundle.get("edge_feature_cols", _parse_edge_feature_cols(train_config.get("edge_feature_cols", "")))
    edge_feature_mean = train_bundle.get("edge_feature_mean", [0.0] * len(edge_feature_cols))
    edge_feature_std = train_bundle.get("edge_feature_std", [1.0] * len(edge_feature_cols))

    node_table, node_features, node_labels, _ = load_peak_node_tables(
        args.promoter_path,
        args.re_path,
        feature_cols=feature_cols,
        label_col=train_config.get("label_col", "atac_signal_sum"),
        normalize_features_by_length=bool(train_config.get("normalize_features_by_length", False)),
    )
    keep_mask = node_table["chr"].astype(str).eq(args.chrom).to_numpy()
    keep_idx = torch.from_numpy(keep_mask.nonzero()[0]).long()
    node_table = node_table.loc[keep_mask].reset_index(drop=True)
    node_features = node_features[keep_idx]
    node_labels = node_labels[keep_idx]

    if args.sample_size > 0 and args.sample_size < len(node_table):
        sample_idx = node_table.sample(n=args.sample_size, random_state=args.seed).sort_index().index.to_numpy()
        node_table = node_table.iloc[sample_idx].reset_index(drop=True)
        node_features = node_features[sample_idx]
        node_labels = node_labels[sample_idx]

    print(f"Held-out nodes={len(node_table)} chrom={args.chrom} feature_dim={node_features.shape[1]}")
    edge_table_path = args.edge_table if args.edge_table else args.abc_edges
    edge_index, edge_attr, edge_base_score, edge_table = _load_edges_for_scoring(
        edge_table_path=edge_table_path,
        node_table=node_table,
        edge_feature_cols=edge_feature_cols,
        edge_feature_mean=edge_feature_mean,
        edge_feature_std=edge_feature_std,
        abc_score_col=args.abc_score_col,
    )
    print(f"Held-out scoring edges={edge_index.shape[1]} edge_feature_dim={edge_attr.shape[1]}")

    device = torch.device(args.device)
    model_mode = train_config.get("model_mode", "dense-idgl")
    if model_mode == "edge-rerank":
        model = EdgeResidualReranker(
            num_features=node_features.shape[1],
            hidden_dim=int(train_config.get("hidden_dim", 128)),
            dropout=float(train_config.get("dropout", 0.2)),
            edge_feature_dim=len(edge_feature_cols),
            abc_logit_scale=float(train_config.get("abc_logit_scale", 1.0)),
            delta_logit_scale=float(train_config.get("delta_logit_scale", 0.25)),
        ).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()

        node_features = node_features.to(device)
        node_labels = node_labels.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        edge_base_score = edge_base_score.to(device)

        with torch.no_grad():
            output = model(
                node_features,
                edge_index,
                edge_attr=edge_attr,
                edge_base_score=edge_base_score,
                return_output=True,
            )

        edge_score_table = build_edge_score_table(
            edge_table,
            edge_logits=output.edge_logits,
            edge_delta_logits=output.edge_delta_logits,
            base_scores=edge_base_score,
        )
        edge_score_path = output_dir / "edge_scores.tsv"
        edge_score_table.to_csv(edge_score_path, sep="\t", index=False)

        edge_supervised_adj = None
        if 0 < len(node_table) <= args.max_dense_output_nodes:
            edge_supervised_adj = build_edge_score_adj(len(node_table), edge_index, output.edge_logits)

        torch.save(
            {
                "model_mode": "edge-rerank",
                "optimized_adj": None,
                "edge_supervised_adj": edge_supervised_adj.detach().cpu() if edge_supervised_adj is not None else None,
                "node_pred": None,
                "node_labels": node_labels.detach().cpu(),
                "feature_cols": feature_cols,
                "node_table": node_table,
                "adj_source": "abc_residual_edge_table",
                "abc_edges": args.abc_edges,
                "abc_score_col": args.abc_score_col,
                "edge_labels": args.edge_table,
                "edge_feature_cols": edge_feature_cols,
                "edge_feature_mean": edge_feature_mean,
                "edge_feature_std": edge_feature_std,
                "edge_score_table_path": str(edge_score_path),
                "abc_logit_scale": float(train_config.get("abc_logit_scale", 1.0)),
                "delta_logit_scale": float(train_config.get("delta_logit_scale", 0.25)),
                "heldout_chrom": args.chrom,
                "train_bundle": args.train_bundle,
            },
            output_dir / "ep_idgl_outputs.pt",
        )
        with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        print(f"Saved held-out edge scores to {edge_score_path}")
        print(f"Saved held-out outputs to {output_dir / 'ep_idgl_outputs.pt'}")
        return

    abc_edges = load_abc_edges(args.abc_edges)
    init_adj = build_candidate_adj_from_abc_edges(
        node_table=node_table,
        abc_edges=abc_edges,
        score_col=args.abc_score_col,
        symmetric=True,
        include_self=True,
    )

    model = PeakLevelIDGLPyG(
        num_features=node_features.shape[1],
        hidden_dim=int(train_config.get("hidden_dim", 128)),
        num_layers=int(train_config.get("num_layers", 2)),
        dropout=float(train_config.get("dropout", 0.2)),
        graph_alpha=float(train_config.get("graph_alpha", 0.5)),
        topk_edges=int(train_config.get("topk_edges", 20)),
        graph_iters=int(train_config.get("graph_iters", 1)),
        edge_feature_dim=len(edge_feature_cols),
        return_dense_adj=True,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    node_features = node_features.to(device)
    node_labels = node_labels.to(device)
    init_adj = init_adj.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)

    with torch.no_grad():
        output = model(
            init_adj,
            node_features,
            node_labels,
            edge_index=edge_index,
            edge_attr=edge_attr,
            return_output=True,
        )
        edge_supervised_adj = build_edge_score_adj(len(node_table), edge_index, output.edge_logits)

    torch.save(
        {
            "optimized_adj": output.optimized_adj.detach().cpu(),
            "edge_supervised_adj": edge_supervised_adj.detach().cpu(),
            "node_pred": output.node_pred.detach().cpu(),
            "node_labels": node_labels.detach().cpu(),
            "feature_cols": feature_cols,
            "node_table": node_table,
            "adj_source": "abc",
            "abc_edges": args.abc_edges,
            "abc_score_col": args.abc_score_col,
            "edge_labels": args.edge_table,
            "edge_feature_cols": edge_feature_cols,
            "edge_feature_mean": edge_feature_mean,
            "edge_feature_std": edge_feature_std,
            "heldout_chrom": args.chrom,
            "train_bundle": args.train_bundle,
        },
        output_dir / "ep_idgl_outputs.pt",
    )
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"Saved held-out outputs to {output_dir / 'ep_idgl_outputs.pt'}")


if __name__ == "__main__":
    main()
