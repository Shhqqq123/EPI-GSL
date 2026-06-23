from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_utils import build_candidate_adj_from_abc_edges, build_candidate_adj_from_distance, load_abc_edges, load_hic_bedpe
from eval_utils import (
    compare_topk_edges,
    evaluate_edge_label_ranking,
    evaluate_with_hic,
    filter_bedpe_to_node_chroms,
    load_edge_label_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate optimized graph against Hi-C bedpe")
    parser.add_argument("--outputs-path", type=str, default=str(CURRENT_DIR.parent / "outputs" / "epi_gsl" / "ep_idgl_outputs.pt"))
    parser.add_argument("--hic-bedpe", type=str, default=str(CURRENT_DIR.parent / "ENCFF308MMM.bedpe" / "ENCFF308MMM.bedpe"))
    parser.add_argument("--max-distance", type=int, default=200000)
    parser.add_argument("--abc-edges", type=str, default="", help="Optional ABC edge table TSV used to rebuild initial adjacency.")
    parser.add_argument("--abc-score-col", type=str, default="abc_score")
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--edge-labels", type=str, default="", help="Optional edge label TSV used for AUROC/AUPRC ranking metrics.")
    parser.add_argument("--edge-label-col", type=str, default="edge_label_test", help="Column in --edge-labels used for ranking evaluation.")
    parser.add_argument("--edge-metric-topks", type=str, default="500,1000,2000", help="Comma-separated K values for precision/recall at K.")
    parser.add_argument("--hic-split-name", type=str, default="all", help="Label printed for the Hi-C BEDPE split being evaluated.")
    parser.add_argument("--ep-only", action="store_true")
    return parser.parse_args()


def _parse_topks(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _print_edge_label_metrics(name: str, metrics: dict[str, float], topks: list[int]) -> None:
    print(f"{name} edge-label ranking metrics:")
    print(
        f"edges={int(metrics['num_edges'])} "
        f"positives={int(metrics['positive_edges'])} "
        f"prevalence={metrics['prevalence']:.6f} "
        f"auroc={metrics['auroc']:.6f} "
        f"auprc={metrics['auprc']:.6f}"
    )
    for k in topks:
        hit_key = f"hit_count_at_{k}"
        precision_key = f"precision_at_{k}"
        recall_key = f"recall_at_{k}"
        enrichment_key = f"enrichment_at_{k}"
        if hit_key not in metrics:
            continue
        print(
            f"  @{k}: "
            f"hit_count={int(metrics[hit_key])} "
            f"precision={metrics[precision_key]:.6f} "
            f"recall={metrics[recall_key]:.6f} "
            f"enrichment={metrics[enrichment_key]:.3f}"
        )


def main() -> None:
    args = parse_args()
    print(f"Loading bundle: {args.outputs_path}")
    bundle = torch.load(args.outputs_path, map_location="cpu")
    print("Bundle loaded.")
    node_table = bundle["node_table"]
    optimized_adj = bundle["optimized_adj"]
    edge_supervised_adj = bundle.get("edge_supervised_adj")

    if args.abc_edges:
        print(f"Rebuilding initial adjacency from ABC edges: {args.abc_edges}")
        abc_edges = load_abc_edges(args.abc_edges)
        init_adj = build_candidate_adj_from_abc_edges(
            node_table=node_table,
            abc_edges=abc_edges,
            score_col=args.abc_score_col,
            symmetric=True,
            include_self=True,
        )
    else:
        print("Rebuilding initial adjacency from genomic distance ...")
        init_adj = build_candidate_adj_from_distance(
            node_table=node_table,
            max_distance=args.max_distance,
            same_chrom_only=True,
            ep_only=args.ep_only,
            symmetric=True,
            normalize=True,
        )
    print("Initial adjacency rebuilt.")

    print(f"Loading Hi-C bedpe ({args.hic_split_name}): {args.hic_bedpe}")
    hic_df = load_hic_bedpe(args.hic_bedpe)
    filtered_hic_df = filter_bedpe_to_node_chroms(hic_df, node_table)
    node_chroms = ",".join(sorted(node_table["chr"].astype(str).unique()))
    print(f"Node chromosomes: {node_chroms}")
    print(f"Hi-C loops before chrom filter: {len(hic_df)}")
    print(f"Hi-C loops after chrom filter: {len(filtered_hic_df)}")
    print("Evaluating initial graph ...")
    init_metrics = evaluate_with_hic(init_adj, node_table, filtered_hic_df, topk=args.topk)
    print("Evaluating optimized graph ...")
    opt_metrics = evaluate_with_hic(optimized_adj, node_table, filtered_hic_df, topk=args.topk)
    rank_metrics = compare_topk_edges(init_adj, optimized_adj, topk=args.topk)
    edge_metrics = None
    edge_rank_metrics = None
    if edge_supervised_adj is not None:
        print("Evaluating edge-supervised graph ...")
        edge_metrics = evaluate_with_hic(edge_supervised_adj, node_table, filtered_hic_df, topk=args.topk)
        edge_rank_metrics = compare_topk_edges(init_adj, edge_supervised_adj, topk=args.topk)

    print("Initial graph metrics:")
    print(
        f"topk={int(init_metrics['topk'])} "
        f"hit_count={int(init_metrics['hit_count'])} "
        f"hit_rate={init_metrics['hit_rate']:.6f}"
    )
    print("Optimized graph metrics:")
    print(
        f"topk={int(opt_metrics['topk'])} "
        f"hit_count={int(opt_metrics['hit_count'])} "
        f"hit_rate={opt_metrics['hit_rate']:.6f}"
    )
    print(f"Delta hit_rate={opt_metrics['hit_rate'] - init_metrics['hit_rate']:.6f}")
    if edge_metrics is not None:
        print("Edge-supervised graph metrics:")
        print(
            f"topk={int(edge_metrics['topk'])} "
            f"hit_count={int(edge_metrics['hit_count'])} "
            f"hit_rate={edge_metrics['hit_rate']:.6f}"
        )
        print(f"Edge-supervised delta hit_rate={edge_metrics['hit_rate'] - init_metrics['hit_rate']:.6f}")
    print("Initial vs optimized ranking diagnostics:")
    print(
        f"topk_overlap={rank_metrics['topk_overlap']:.6f} "
        f"topk_jaccard={rank_metrics['topk_jaccard']:.6f} "
        f"init_topk_size={int(rank_metrics['init_topk_size'])} "
        f"opt_topk_size={int(rank_metrics['opt_topk_size'])}"
    )
    if edge_rank_metrics is not None:
        print("Initial vs edge-supervised ranking diagnostics:")
        print(
            f"topk_overlap={edge_rank_metrics['topk_overlap']:.6f} "
            f"topk_jaccard={edge_rank_metrics['topk_jaccard']:.6f} "
            f"init_topk_size={int(edge_rank_metrics['init_topk_size'])} "
            f"edge_topk_size={int(edge_rank_metrics['opt_topk_size'])}"
        )

    if args.edge_labels:
        topks = _parse_topks(args.edge_metric_topks)
        print(f"Loading edge labels for ranking metrics: {args.edge_labels}")
        edge_label_table = load_edge_label_table(args.edge_labels, node_table, label_col=args.edge_label_col)
        print(f"Edge label column: {args.edge_label_col}")
        init_edge_label_metrics = evaluate_edge_label_ranking(init_adj, edge_label_table, topks=topks)
        opt_edge_label_metrics = evaluate_edge_label_ranking(optimized_adj, edge_label_table, topks=topks)
        _print_edge_label_metrics("Initial graph", init_edge_label_metrics, topks)
        _print_edge_label_metrics("Optimized graph", opt_edge_label_metrics, topks)
        if edge_supervised_adj is not None:
            edge_sup_label_metrics = evaluate_edge_label_ranking(edge_supervised_adj, edge_label_table, topks=topks)
            _print_edge_label_metrics("Edge-supervised graph", edge_sup_label_metrics, topks)


if __name__ == "__main__":
    main()
