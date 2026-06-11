from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_utils import build_candidate_adj_from_abc_edges, build_candidate_adj_from_distance, load_abc_edges, load_hic_bedpe
from eval_utils import compare_topk_edges, evaluate_with_hic, filter_bedpe_to_node_chroms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate optimized graph against Hi-C bedpe")
    parser.add_argument("--outputs-path", type=str, default=str(CURRENT_DIR.parent / "outputs" / "epi_gsl" / "ep_idgl_outputs.pt"))
    parser.add_argument("--hic-bedpe", type=str, default=str(CURRENT_DIR.parent / "ENCFF308MMM.bedpe" / "ENCFF308MMM.bedpe"))
    parser.add_argument("--max-distance", type=int, default=200000)
    parser.add_argument("--abc-edges", type=str, default="", help="Optional ABC edge table TSV used to rebuild initial adjacency.")
    parser.add_argument("--abc-score-col", type=str, default="abc_score")
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--ep-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading bundle: {args.outputs_path}")
    bundle = torch.load(args.outputs_path, map_location="cpu")
    print("Bundle loaded.")
    node_table = bundle["node_table"]
    optimized_adj = bundle["optimized_adj"]

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

    print(f"Loading Hi-C bedpe: {args.hic_bedpe}")
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
    print("Initial vs optimized ranking diagnostics:")
    print(
        f"topk_overlap={rank_metrics['topk_overlap']:.6f} "
        f"topk_jaccard={rank_metrics['topk_jaccard']:.6f} "
        f"init_topk_size={int(rank_metrics['init_topk_size'])} "
        f"opt_topk_size={int(rank_metrics['opt_topk_size'])}"
    )


if __name__ == "__main__":
    main()
