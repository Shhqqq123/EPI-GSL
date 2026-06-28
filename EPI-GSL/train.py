from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_utils import (
    build_candidate_adj_from_abc_edges,
    build_candidate_adj_from_distance,
    load_abc_edges,
    load_peak_node_tables,
)
from loss import PeakLevelIDGLLoss
from model import EdgeResidualReranker, PeakLevelIDGLPyG


@dataclass
class EdgeSupervisionBundle:
    all_edge_index: torch.Tensor
    all_edge_attr: torch.Tensor
    all_edge_labels: torch.Tensor
    all_edge_base_scores: torch.Tensor
    train_edge_index: torch.Tensor
    train_edge_attr: torch.Tensor
    train_edge_labels: torch.Tensor
    train_edge_base_scores: torch.Tensor
    train_pos_count: int
    edge_feature_mean: np.ndarray
    edge_feature_std: np.ndarray
    edge_table: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train peak-level IDGL in modular form")
    parser.add_argument(
        "--model-mode",
        choices=["edge-rerank", "dense-idgl"],
        default="edge-rerank",
        help="edge-rerank scales on ABC candidate edges; dense-idgl keeps the original dense graph learner.",
    )
    parser.add_argument("--promoter-path", type=str, default=str(CURRENT_DIR.parent / "promoter_nodes_full.tsv"))
    parser.add_argument("--re-path", type=str, default=str(CURRENT_DIR.parent / "re_nodes_full.tsv"))
    parser.add_argument("--output-dir", type=str, default=str(CURRENT_DIR.parent / "outputs" / "epi_gsl"))
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--max-distance", type=int, default=200000)
    parser.add_argument("--abc-edges", type=str, default="", help="Optional ABC edge table TSV.")
    parser.add_argument("--abc-score-col", type=str, default="abc_score")
    parser.add_argument("--label-col", type=str, default="atac_signal_sum")
    parser.add_argument("--normalize-features-by-length", action="store_true")
    parser.add_argument("--chrom", type=str, default="")
    parser.add_argument("--include-chroms", type=str, default="", help="Comma-separated chromosomes included for training, for example chr1,chr2.")
    parser.add_argument("--exclude-chroms", type=str, default="", help="Comma-separated chromosomes excluded from training, for example chr5,chrX,chrY.")
    parser.add_argument("--ep-only", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--graph-alpha", type=float, default=0.5)
    parser.add_argument("--topk-edges", type=int, default=20)
    parser.add_argument("--graph-iters", type=int, default=1, help="Number of iterative graph learning/message passing rounds.")
    parser.add_argument("--edge-labels", type=str, default="", help="Optional edge label TSV from make_edge_labels.py.")
    parser.add_argument("--edge-label-col", type=str, default="edge_label", help="Column in --edge-labels used as training target.")
    parser.add_argument("--edge-loss-weight", type=float, default=1.0, help="Weight for supervised edge BCE loss.")
    parser.add_argument("--edge-feature-cols", type=str, default="abc_score,distance")
    parser.add_argument("--max-edge-train-samples", type=int, default=0, help="Optional cap for supervised edge samples.")
    parser.add_argument("--negative-ratio", type=float, default=5.0, help="Negatives per positive for edge supervision.")
    parser.add_argument(
        "--negative-sampling",
        choices=["random", "distance-matched", "abc-distance-matched"],
        default="abc-distance-matched",
        help="How to sample reliable negatives for edge supervision.",
    )
    parser.add_argument("--negative-distance-bins", type=int, default=20)
    parser.add_argument("--negative-abc-bins", type=int, default=10)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.1)
    parser.add_argument("--ranking-margin", type=float, default=1.0)
    parser.add_argument("--ranking-negatives-per-positive", type=int, default=2)
    parser.add_argument("--ranking-max-pairs", type=int, default=20000)
    parser.add_argument("--delta-l2-weight", type=float, default=1e-3)
    parser.add_argument("--abc-logit-scale", type=float, default=1.0)
    parser.add_argument(
        "--delta-logit-scale",
        type=float,
        default=0.25,
        help="Scale applied to learned residual logits before adding them to ABC logits.",
    )
    parser.add_argument(
        "--max-dense-output-nodes",
        type=int,
        default=8000,
        help="Only materialize dense score adjacency for edge-rerank outputs up to this many nodes.",
    )
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--sparsity-weight", type=float, default=1e-3)
    parser.add_argument("--smooth-weight", type=float, default=1e-2)
    parser.add_argument("--stability-weight", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_edge_feature_cols(raw: str) -> List[str]:
    return [col.strip() for col in raw.split(",") if col.strip()]


def _edge_feature_frame(edge_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    features: Dict[str, np.ndarray] = {}
    for col in feature_cols:
        if col not in edge_df.columns:
            raise KeyError(f"Missing edge feature column: {col}")
        values = pd.to_numeric(edge_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if col.lower() in {"distance", "dist"}:
            values = np.log1p(np.clip(values, a_min=0.0, a_max=None)).astype(np.float32)
        features[col] = values
    return pd.DataFrame(features)


def _parse_chrom_list(raw: str) -> List[str]:
    return [chrom.strip() for chrom in raw.split(",") if chrom.strip()]


def _standardize_edge_features(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.size == 0:
        width = values.shape[1] if values.ndim == 2 else 0
        return values.astype(np.float32), np.zeros(width, dtype=np.float32), np.ones(width, dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.where(std <= 1e-6, 1.0, std)
    return ((values - mean) / std).astype(np.float32), mean.squeeze(0).astype(np.float32), std.squeeze(0).astype(np.float32)


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins <= 1 or len(values) == 0:
        return np.zeros(len(values), dtype=np.int64)
    ranks = pd.Series(values).rank(method="first", pct=True).to_numpy(dtype=np.float64)
    bins = np.floor(ranks * n_bins).astype(np.int64)
    return np.clip(bins, 0, n_bins - 1)


def _sample_edge_train_indices(
    edge_df: pd.DataFrame,
    labels: np.ndarray,
    negative_ratio: float,
    max_train_samples: int,
    seed: int,
    negative_sampling: str,
    distance_bins: int,
    abc_bins: int,
    abc_score_col: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(labels > 0.5)
    neg_idx = np.flatnonzero(labels <= 0.5)
    if len(pos_idx) == 0 or len(neg_idx) == 0 or negative_ratio <= 0:
        train_idx = np.arange(len(labels))
    elif negative_sampling == "random" or "distance" not in edge_df.columns:
        neg_keep = min(len(neg_idx), int(np.ceil(len(pos_idx) * negative_ratio)))
        sampled_neg = rng.choice(neg_idx, size=neg_keep, replace=False) if neg_keep < len(neg_idx) else neg_idx
        train_idx = np.concatenate([pos_idx, sampled_neg])
    else:
        distances = pd.to_numeric(edge_df["distance"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        dist_key = _quantile_bins(np.log1p(np.clip(distances, a_min=0.0, a_max=None)), distance_bins)
        if negative_sampling == "abc-distance-matched" and abc_score_col in edge_df.columns:
            abc_scores = pd.to_numeric(edge_df[abc_score_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            abc_key = _quantile_bins(abc_scores, abc_bins)
            keys = list(zip(dist_key.tolist(), abc_key.tolist()))
        else:
            keys = dist_key.tolist()

        selected_neg: set[int] = set()
        target_neg = min(len(neg_idx), int(np.ceil(len(pos_idx) * negative_ratio)))
        neg_by_key: Dict[object, List[int]] = {}
        pos_by_key: Dict[object, int] = {}
        for idx in neg_idx:
            neg_by_key.setdefault(keys[int(idx)], []).append(int(idx))
        for idx in pos_idx:
            pos_by_key[keys[int(idx)]] = pos_by_key.get(keys[int(idx)], 0) + 1

        for key, pos_count in pos_by_key.items():
            candidates = neg_by_key.get(key, [])
            if not candidates:
                continue
            keep = min(len(candidates), int(np.ceil(pos_count * negative_ratio)))
            chosen = rng.choice(candidates, size=keep, replace=False) if keep < len(candidates) else np.asarray(candidates)
            selected_neg.update(int(i) for i in chosen)

        if len(selected_neg) < target_neg:
            remaining = np.asarray([idx for idx in neg_idx if int(idx) not in selected_neg], dtype=np.int64)
            need = min(len(remaining), target_neg - len(selected_neg))
            if need > 0:
                chosen = rng.choice(remaining, size=need, replace=False) if need < len(remaining) else remaining
                selected_neg.update(int(i) for i in chosen)

        train_idx = np.concatenate([pos_idx, np.asarray(sorted(selected_neg), dtype=np.int64)])

    if max_train_samples > 0 and len(train_idx) > max_train_samples:
        pos_idx = train_idx[labels[train_idx] > 0.5]
        neg_idx = train_idx[labels[train_idx] <= 0.5]
        max_pos = min(len(pos_idx), max_train_samples)
        if len(pos_idx) > max_pos:
            pos_idx = rng.choice(pos_idx, size=max_pos, replace=False)
        remaining = max(0, max_train_samples - len(pos_idx))
        if len(neg_idx) > remaining:
            neg_idx = rng.choice(neg_idx, size=remaining, replace=False)
        train_idx = np.concatenate([pos_idx, neg_idx])

    rng.shuffle(train_idx)
    return train_idx.astype(np.int64)


def pairwise_ranking_loss(
    edge_logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
    negatives_per_positive: int,
    max_pairs: int,
) -> torch.Tensor:
    pos_idx = torch.nonzero(labels > 0.5, as_tuple=False).view(-1)
    neg_idx = torch.nonzero(labels <= 0.5, as_tuple=False).view(-1)
    if pos_idx.numel() == 0 or neg_idx.numel() == 0 or max_pairs <= 0 or negatives_per_positive <= 0:
        return edge_logits.new_tensor(0.0)
    pair_count = min(max_pairs, int(pos_idx.numel()) * int(negatives_per_positive))
    pos_choice = pos_idx[torch.randint(pos_idx.numel(), (pair_count,), device=edge_logits.device)]
    neg_choice = neg_idx[torch.randint(neg_idx.numel(), (pair_count,), device=edge_logits.device)]
    return F.relu(margin - (edge_logits[pos_choice] - edge_logits[neg_choice])).mean()


def load_edge_supervision(
    edge_label_path: str,
    node_table: pd.DataFrame,
    edge_feature_cols: List[str],
    edge_label_col: str,
    abc_score_col: str,
    negative_ratio: float,
    max_train_samples: int,
    seed: int,
    negative_sampling: str,
    negative_distance_bins: int,
    negative_abc_bins: int,
) -> EdgeSupervisionBundle:
    edge_df = pd.read_csv(edge_label_path, sep="\t").copy()
    required = ["re_node_id", "promoter_node_id", edge_label_col, abc_score_col]
    missing = [col for col in required if col not in edge_df.columns]
    if missing:
        raise KeyError(f"Edge label table is missing required columns: {missing}")

    node_to_idx = {str(node_id): idx for idx, node_id in enumerate(node_table["node_id"].astype(str))}
    src_idx = edge_df["re_node_id"].astype(str).map(node_to_idx)
    dst_idx = edge_df["promoter_node_id"].astype(str).map(node_to_idx)
    keep = src_idx.notna() & dst_idx.notna()
    edge_df = edge_df.loc[keep].reset_index(drop=True)
    src = src_idx.loc[keep].astype(int).to_numpy(dtype=np.int64)
    dst = dst_idx.loc[keep].astype(int).to_numpy(dtype=np.int64)
    labels = pd.to_numeric(edge_df[edge_label_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    if len(labels) == 0:
        empty_index = torch.empty((2, 0), dtype=torch.long)
        empty_float = torch.empty((0,), dtype=torch.float32)
        empty_attr = torch.empty((0, len(edge_feature_cols)), dtype=torch.float32)
        empty_mean = np.zeros(len(edge_feature_cols), dtype=np.float32)
        empty_std = np.ones(len(edge_feature_cols), dtype=np.float32)
        return EdgeSupervisionBundle(
            all_edge_index=empty_index,
            all_edge_attr=empty_attr,
            all_edge_labels=empty_float,
            all_edge_base_scores=empty_float,
            train_edge_index=empty_index,
            train_edge_attr=empty_attr,
            train_edge_labels=empty_float,
            train_edge_base_scores=empty_float,
            train_pos_count=0,
            edge_feature_mean=empty_mean,
            edge_feature_std=empty_std,
            edge_table=edge_df,
        )

    edge_features = _edge_feature_frame(edge_df, edge_feature_cols).to_numpy(dtype=np.float32)
    edge_features, edge_feature_mean, edge_feature_std = _standardize_edge_features(edge_features)
    base_scores = pd.to_numeric(edge_df[abc_score_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    train_idx = _sample_edge_train_indices(
        edge_df=edge_df,
        labels=labels,
        negative_ratio=negative_ratio,
        max_train_samples=max_train_samples,
        seed=seed,
        negative_sampling=negative_sampling,
        distance_bins=negative_distance_bins,
        abc_bins=negative_abc_bins,
        abc_score_col=abc_score_col,
    )

    all_edge_index = torch.from_numpy(np.vstack([src, dst])).long()
    all_edge_attr = torch.from_numpy(edge_features).float()
    train_edge_index = all_edge_index[:, train_idx]
    train_edge_attr = all_edge_attr[train_idx]
    train_labels = torch.from_numpy(labels[train_idx]).float()
    pos_count = int((train_labels > 0.5).sum().item())
    return EdgeSupervisionBundle(
        all_edge_index=all_edge_index,
        all_edge_attr=all_edge_attr,
        all_edge_labels=torch.from_numpy(labels).float(),
        all_edge_base_scores=torch.from_numpy(base_scores).float(),
        train_edge_index=train_edge_index,
        train_edge_attr=train_edge_attr,
        train_edge_labels=train_labels,
        train_edge_base_scores=torch.from_numpy(base_scores[train_idx]).float(),
        train_pos_count=pos_count,
        edge_feature_mean=edge_feature_mean,
        edge_feature_std=edge_feature_std,
        edge_table=edge_df,
    )


def build_edge_score_adj(num_nodes: int, edge_index: torch.Tensor, edge_logits: torch.Tensor) -> torch.Tensor:
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=edge_logits.device)
    if edge_index.numel() == 0:
        adj.fill_diagonal_(1.0)
        return adj
    scores = torch.sigmoid(edge_logits).float()
    src = edge_index[0].to(edge_logits.device)
    dst = edge_index[1].to(edge_logits.device)
    adj[src, dst] = scores
    adj[dst, src] = scores
    adj.fill_diagonal_(1.0)
    return adj


def build_edge_score_table(
    edge_table: pd.DataFrame,
    edge_logits: torch.Tensor,
    edge_delta_logits: torch.Tensor,
    base_scores: torch.Tensor,
) -> pd.DataFrame:
    result = edge_table.copy()
    result["abc_base_score"] = base_scores.detach().cpu().numpy().astype(np.float32)
    result["edge_delta_logit"] = edge_delta_logits.detach().cpu().numpy().astype(np.float32)
    result["edge_logit"] = edge_logits.detach().cpu().numpy().astype(np.float32)
    result["final_score"] = torch.sigmoid(edge_logits.detach().cpu()).numpy().astype(np.float32)
    return result


def save_edge_rerank_outputs(
    output_dir: Path,
    model: EdgeResidualReranker,
    node_table: pd.DataFrame,
    node_features: torch.Tensor,
    node_labels: torch.Tensor,
    feature_cols: List[str],
    edge_supervision: EdgeSupervisionBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    model.eval()
    with torch.no_grad():
        output = model(
            node_features,
            edge_supervision.all_edge_index.to(device),
            edge_attr=edge_supervision.all_edge_attr.to(device),
            edge_base_score=edge_supervision.all_edge_base_scores.to(device),
            return_output=True,
        )

    edge_score_table = build_edge_score_table(
        edge_supervision.edge_table,
        edge_logits=output.edge_logits,
        edge_delta_logits=output.edge_delta_logits,
        base_scores=edge_supervision.all_edge_base_scores,
    )
    edge_score_path = output_dir / "edge_scores.tsv"
    edge_score_table.to_csv(edge_score_path, sep="\t", index=False)

    edge_supervised_adj = None
    if 0 < len(node_table) <= args.max_dense_output_nodes:
        edge_supervised_adj = build_edge_score_adj(
            len(node_table),
            edge_supervision.all_edge_index.to(output.edge_logits.device),
            output.edge_logits,
        )

    torch.save(model.state_dict(), output_dir / "ep_idgl_model.pt")
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
            "edge_labels": args.edge_labels,
            "edge_label_col": args.edge_label_col,
            "edge_feature_cols": _parse_edge_feature_cols(args.edge_feature_cols),
            "edge_feature_mean": edge_supervision.edge_feature_mean.tolist(),
            "edge_feature_std": edge_supervision.edge_feature_std.tolist(),
            "edge_score_table_path": str(edge_score_path),
            "abc_logit_scale": args.abc_logit_scale,
            "delta_logit_scale": args.delta_logit_scale,
        },
        output_dir / "ep_idgl_outputs.pt",
    )
    print(f"saved model to {output_dir / 'ep_idgl_model.pt'}")
    print(f"saved edge scores to {edge_score_path}")
    print(f"saved outputs to {output_dir / 'ep_idgl_outputs.pt'}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    node_table, node_features, node_labels, feature_cols = load_peak_node_tables(
        args.promoter_path,
        args.re_path,
        label_col=args.label_col,
        normalize_features_by_length=args.normalize_features_by_length,
    )

    print(
        f"Loaded nodes={len(node_table)} features={len(feature_cols)} "
        f"label_col={args.label_col} normalize_features_by_length={args.normalize_features_by_length}"
    )
    print(
        "Label stats before filtering/sample: "
        f"mean={node_labels.float().mean().item():.4f} "
        f"std={node_labels.float().std(unbiased=False).item():.4f}"
    )

    if args.chrom:
        keep_mask = node_table["chr"].astype(str).eq(args.chrom).to_numpy()
        keep_idx = torch.from_numpy(keep_mask.nonzero()[0]).long()
        node_table = node_table.loc[keep_mask].reset_index(drop=True)
        node_features = node_features[keep_idx]
        node_labels = node_labels[keep_idx]
    if args.include_chroms:
        included = set(_parse_chrom_list(args.include_chroms))
        keep_mask = node_table["chr"].astype(str).isin(included).to_numpy()
        keep_idx = torch.from_numpy(keep_mask.nonzero()[0]).long()
        node_table = node_table.loc[keep_mask].reset_index(drop=True)
        node_features = node_features[keep_idx]
        node_labels = node_labels[keep_idx]
    if args.exclude_chroms:
        excluded = set(_parse_chrom_list(args.exclude_chroms))
        keep_mask = ~node_table["chr"].astype(str).isin(excluded).to_numpy()
        keep_idx = torch.from_numpy(keep_mask.nonzero()[0]).long()
        node_table = node_table.loc[keep_mask].reset_index(drop=True)
        node_features = node_features[keep_idx]
        node_labels = node_labels[keep_idx]

    if args.sample_size > 0 and args.sample_size < len(node_table):
        sample_idx = node_table.sample(n=args.sample_size, random_state=args.seed).sort_index().index.to_numpy()
        node_table = node_table.iloc[sample_idx].reset_index(drop=True)
        node_features = node_features[sample_idx]
        node_labels = node_labels[sample_idx]

    print(
        f"Training nodes={len(node_table)} feature_dim={node_features.shape[1]} "
        f"label_mean={node_labels.float().mean().item():.4f} "
        f"label_std={node_labels.float().std(unbiased=False).item():.4f}"
    )

    edge_feature_cols = _parse_edge_feature_cols(args.edge_feature_cols)
    edge_supervision = None
    edge_feature_dim = 0
    if args.edge_labels:
        print(f"Loading edge supervision from: {args.edge_labels}")
        edge_supervision = load_edge_supervision(
            edge_label_path=args.edge_labels,
            node_table=node_table,
            edge_feature_cols=edge_feature_cols,
            edge_label_col=args.edge_label_col,
            abc_score_col=args.abc_score_col,
            negative_ratio=args.negative_ratio,
            max_train_samples=args.max_edge_train_samples,
            seed=args.seed,
            negative_sampling=args.negative_sampling,
            negative_distance_bins=args.negative_distance_bins,
            negative_abc_bins=args.negative_abc_bins,
        )
        edge_feature_dim = edge_supervision.all_edge_attr.shape[1]
        print(
            f"Supervised edges total={edge_supervision.all_edge_index.shape[1]} "
            f"train={edge_supervision.train_edge_index.shape[1]} "
            f"train_pos={edge_supervision.train_pos_count} label_col={args.edge_label_col} "
            f"edge_feature_dim={edge_feature_dim} negative_sampling={args.negative_sampling}"
        )
        if edge_supervision.all_edge_index.shape[1] == 0:
            edge_supervision = None
            print("No supervised edges matched the sampled node table; edge loss disabled.")

    if args.model_mode == "edge-rerank":
        if edge_supervision is None:
            raise ValueError("--model-mode edge-rerank requires --edge-labels with matching candidate edges.")
        device = torch.device(args.device)
        node_features = node_features.to(device)
        node_labels = node_labels.to(device)
        edge_supervision.all_edge_index = edge_supervision.all_edge_index.to(device)
        edge_supervision.all_edge_attr = edge_supervision.all_edge_attr.to(device)
        edge_supervision.all_edge_labels = edge_supervision.all_edge_labels.to(device)
        edge_supervision.all_edge_base_scores = edge_supervision.all_edge_base_scores.to(device)
        edge_supervision.train_edge_index = edge_supervision.train_edge_index.to(device)
        edge_supervision.train_edge_attr = edge_supervision.train_edge_attr.to(device)
        edge_supervision.train_edge_labels = edge_supervision.train_edge_labels.to(device)
        edge_supervision.train_edge_base_scores = edge_supervision.train_edge_base_scores.to(device)

        model = EdgeResidualReranker(
            num_features=node_features.shape[1],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            edge_feature_dim=edge_feature_dim,
            abc_logit_scale=args.abc_logit_scale,
            delta_logit_scale=args.delta_logit_scale,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            output = model(
                node_features,
                edge_supervision.train_edge_index,
                edge_attr=edge_supervision.train_edge_attr,
                edge_base_score=edge_supervision.train_edge_base_scores,
                return_output=True,
            )
            edge_bce = F.binary_cross_entropy_with_logits(output.edge_logits, edge_supervision.train_edge_labels)
            rank_loss = pairwise_ranking_loss(
                output.edge_logits,
                edge_supervision.train_edge_labels,
                margin=args.ranking_margin,
                negatives_per_positive=args.ranking_negatives_per_positive,
                max_pairs=args.ranking_max_pairs,
            )
            delta_l2 = output.edge_delta_logits.pow(2).mean()
            loss = (
                args.edge_loss_weight * edge_bce
                + args.ranking_loss_weight * rank_loss
                + args.delta_l2_weight * delta_l2
            )
            loss.backward()
            optimizer.step()

            record = {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "edge_bce": float(edge_bce.detach().cpu()),
                "ranking_loss": float(rank_loss.detach().cpu()),
                "delta_l2": float(delta_l2.detach().cpu()),
            }
            history.append(record)
            print(
                f"epoch={epoch:03d} "
                f"loss={record['loss']:.4f} "
                f"edge_bce={record['edge_bce']:.4f} "
                f"ranking={record['ranking_loss']:.4f} "
                f"delta_l2={record['delta_l2']:.4f}"
            )

        save_edge_rerank_outputs(
            output_dir=output_dir,
            model=model,
            node_table=node_table,
            node_features=node_features,
            node_labels=node_labels,
            feature_cols=feature_cols,
            edge_supervision=edge_supervision,
            args=args,
            device=device,
        )
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        print(f"saved metrics to {output_dir / 'metrics.json'}")
        print(f"saved config to {output_dir / 'run_config.json'}")
        return

    if args.abc_edges:
        print(f"Building initial adjacency from ABC edges: {args.abc_edges}")
        abc_edges = load_abc_edges(args.abc_edges)
        adj = build_candidate_adj_from_abc_edges(
            node_table=node_table,
            abc_edges=abc_edges,
            score_col=args.abc_score_col,
            symmetric=True,
            include_self=True,
        )
        adj_source = "abc"
    else:
        print("Building initial adjacency from genomic distance")
        adj = build_candidate_adj_from_distance(
            node_table,
            max_distance=args.max_distance,
            same_chrom_only=True,
            ep_only=args.ep_only,
            symmetric=True,
            normalize=True,
        )
        adj_source = "distance"

    print(f"Initial adj shape={tuple(adj.shape)} nonzero={int((adj > 0).sum().item())}")



    device = torch.device(args.device)
    node_features = node_features.to(device)
    node_labels = node_labels.to(device)
    adj = adj.to(device)

    model = PeakLevelIDGLPyG(
        num_features=node_features.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        graph_alpha=args.graph_alpha,
        topk_edges=args.topk_edges,
        graph_iters=args.graph_iters,
        edge_feature_dim=edge_feature_dim,
        return_dense_adj=True,
    ).to(device)
    loss_fn = PeakLevelIDGLLoss(
        recon_weight=args.recon_weight,
        sparsity_weight=args.sparsity_weight,
        smooth_weight=args.smooth_weight,
        stability_weight=args.stability_weight,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    edge_loss_enabled = edge_supervision is not None and args.edge_loss_weight > 0
    if edge_supervision is not None:
        all_edge_index = edge_supervision.all_edge_index.to(device)
        all_edge_attr = edge_supervision.all_edge_attr.to(device)
        all_edge_labels = edge_supervision.all_edge_labels.to(device)
        train_edge_index = edge_supervision.train_edge_index.to(device)
        train_edge_attr = edge_supervision.train_edge_attr.to(device)
        train_edge_labels = edge_supervision.train_edge_labels.to(device)
        edge_feature_mean = edge_supervision.edge_feature_mean
        edge_feature_std = edge_supervision.edge_feature_std
    else:
        all_edge_index = all_edge_attr = all_edge_labels = None
        train_edge_index = train_edge_attr = train_edge_labels = None
        edge_feature_mean = np.zeros(edge_feature_dim, dtype=np.float32)
        edge_feature_std = np.ones(edge_feature_dim, dtype=np.float32)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        output = model(
            adj,
            node_features,
            node_labels,
            edge_index=train_edge_index if edge_loss_enabled else None,
            edge_attr=train_edge_attr if edge_loss_enabled else None,
            return_output=True,
        )
        losses = loss_fn(output.optimized_adj, output.node_pred, node_labels, adj)
        if edge_loss_enabled and output.edge_logits is not None:
            edge_bce = F.binary_cross_entropy_with_logits(output.edge_logits, train_edge_labels)
            losses["edge_bce"] = edge_bce.detach()
            losses["loss"] = losses["loss"] + args.edge_loss_weight * edge_bce
        else:
            losses["edge_bce"] = adj.new_tensor(0.0)
        losses["loss"].backward()
        optimizer.step()

        record = {k: float(v.detach().cpu()) for k, v in losses.items()}
        record["epoch"] = epoch
        history.append(record)
        print(
            f"epoch={epoch:03d} "
            f"loss={record['loss']:.4f} "
            f"node_mse={record['node_mse']:.4f} "
            f"sparsity={record['sparsity']:.4f} "
            f"stability={record['stability']:.4f} "
            f"smoothness={record['smoothness']:.4f} "
            f"edge_bce={record['edge_bce']:.4f}"
        )

    model.eval()
    with torch.no_grad():
        output = model(
            adj,
            node_features,
            node_labels,
            edge_index=all_edge_index,
            edge_attr=all_edge_attr,
            return_output=True,
        )
        optimized_adj = output.optimized_adj
        node_pred = output.node_pred
        edge_supervised_adj = None
        if all_edge_index is not None and output.edge_logits is not None:
            edge_supervised_adj = build_edge_score_adj(len(node_table), all_edge_index, output.edge_logits)

    torch.save(model.state_dict(), output_dir / "ep_idgl_model.pt")
    torch.save(
        {
            "model_mode": "dense-idgl",
            "optimized_adj": optimized_adj.detach().cpu(),
            "edge_supervised_adj": edge_supervised_adj.detach().cpu() if edge_supervised_adj is not None else None,
            "node_pred": node_pred.detach().cpu(),
            "node_labels": node_labels.detach().cpu(),
            "feature_cols": feature_cols,
            "node_table": node_table,
            "adj_source": adj_source,
            "abc_edges": args.abc_edges,
            "abc_score_col": args.abc_score_col,
            "edge_labels": args.edge_labels,
            "edge_label_col": args.edge_label_col,
            "edge_feature_cols": edge_feature_cols,
            "edge_feature_mean": edge_feature_mean.tolist(),
            "edge_feature_std": edge_feature_std.tolist(),
        },
        output_dir / "ep_idgl_outputs.pt",
    )
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(f"saved model to {output_dir / 'ep_idgl_model.pt'}")
    print(f"saved outputs to {output_dir / 'ep_idgl_outputs.pt'}")
    print(f"saved config to {output_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
