from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

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
from model import PeakLevelIDGLPyG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train peak-level IDGL in modular form")
    parser.add_argument("--promoter-path", type=str, default=str(CURRENT_DIR.parent / "promoter_nodes_full.tsv"))
    parser.add_argument("--re-path", type=str, default=str(CURRENT_DIR.parent / "re_nodes_full.tsv"))
    parser.add_argument("--output-dir", type=str, default=str(CURRENT_DIR.parent / "outputs" / "epi_gsl"))
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--max-distance", type=int, default=200000)
    parser.add_argument("--abc-edges", type=str, default="", help="Optional ABC edge table TSV.")
    parser.add_argument("--abc-score-col", type=str, default="abc_score")
    parser.add_argument("--chrom", type=str, default="")
    parser.add_argument("--ep-only", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--graph-alpha", type=float, default=0.5)
    parser.add_argument("--topk-edges", type=int, default=20)
    parser.add_argument("--graph-iters", type=int, default=1, help="Number of iterative graph learning/message passing rounds.")
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    node_table, node_features, node_labels, feature_cols = load_peak_node_tables(
        args.promoter_path,
        args.re_path,
    )

    if args.chrom:
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
        return_dense_adj=True,
    ).to(device)
    loss_fn = PeakLevelIDGLLoss(
        recon_weight=args.recon_weight,
        sparsity_weight=args.sparsity_weight,
        smooth_weight=args.smooth_weight,
        stability_weight=args.stability_weight,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        optimized_adj, node_pred = model(adj, node_features, node_labels)
        losses = loss_fn(optimized_adj, node_pred, node_labels, adj)
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
            f"smoothness={record['smoothness']:.4f}"
        )

    model.eval()
    with torch.no_grad():
        optimized_adj, node_pred = model(adj, node_features, node_labels)

    torch.save(model.state_dict(), output_dir / "ep_idgl_model.pt")
    torch.save(
        {
            "optimized_adj": optimized_adj.detach().cpu(),
            "node_pred": node_pred.detach().cpu(),
            "node_labels": node_labels.detach().cpu(),
            "feature_cols": feature_cols,
            "node_table": node_table,
            "adj_source": adj_source,
            "abc_edges": args.abc_edges,
            "abc_score_col": args.abc_score_col,
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
