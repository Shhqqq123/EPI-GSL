from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label ABC candidate edges by overlap with Hi-C BEDPE loops.")
    parser.add_argument("--abc-edges", type=str, required=True, help="ABC edge TSV from make_abc_edges.py.")
    parser.add_argument("--hic-bedpe", type=str, required=True, help="Hi-C BEDPE file.")
    parser.add_argument("--output-path", type=str, required=True, help="Output TSV with edge_label column.")
    parser.add_argument("--chrom", type=str, default="", help="Optional chromosome subset, for example chr5.")
    parser.add_argument(
        "--anchor-slop",
        type=int,
        default=0,
        help="Optional bp padding around both candidate intervals and BEDPE anchors before overlap.",
    )
    return parser.parse_args()


def _read_bedpe(path: str) -> pd.DataFrame:
    bedpe = pd.read_csv(path, sep="\t", comment="#", header=None)
    bedpe = bedpe.iloc[:, :6].copy()
    bedpe.columns = ["chr1", "start1", "end1", "chr2", "start2", "end2"]
    for col in ["start1", "end1", "start2", "end2"]:
        bedpe[col] = pd.to_numeric(bedpe[col], errors="coerce").fillna(0).astype(np.int64)
    bedpe["chr1"] = bedpe["chr1"].astype(str)
    bedpe["chr2"] = bedpe["chr2"].astype(str)
    return bedpe


def _overlap(chr_a: str, start_a: int, end_a: int, chr_b: str, start_b: int, end_b: int) -> bool:
    if chr_a != chr_b:
        return False
    return not (end_a < start_b or end_b < start_a)


def _edge_hits_loop(edge: pd.Series, loop: pd.Series, slop: int) -> bool:
    re_interval = (
        str(edge["chr"]),
        int(edge["re_start"]) - slop,
        int(edge["re_end"]) + slop,
    )
    prom_interval = (
        str(edge["chr"]),
        int(edge["prom_start"]) - slop,
        int(edge["prom_end"]) + slop,
    )
    anchor1 = (
        str(loop["chr1"]),
        int(loop["start1"]) - slop,
        int(loop["end1"]) + slop,
    )
    anchor2 = (
        str(loop["chr2"]),
        int(loop["start2"]) - slop,
        int(loop["end2"]) + slop,
    )

    direct = _overlap(*re_interval, *anchor1) and _overlap(*prom_interval, *anchor2)
    swapped = _overlap(*re_interval, *anchor2) and _overlap(*prom_interval, *anchor1)
    return direct or swapped


def _loop_bounds(loop: pd.Series) -> Tuple[int, int]:
    starts = [int(loop["start1"]), int(loop["start2"])]
    ends = [int(loop["end1"]), int(loop["end2"])]
    return min(starts), max(ends)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    edges = pd.read_csv(args.abc_edges, sep="\t").copy()
    required = ["chr", "re_start", "re_end", "prom_start", "prom_end"]
    missing = [col for col in required if col not in edges.columns]
    if missing:
        raise KeyError(f"ABC edge table is missing required columns: {missing}")

    bedpe = _read_bedpe(args.hic_bedpe)
    if args.chrom:
        edges = edges[edges["chr"].astype(str).eq(args.chrom)].copy()
        bedpe = bedpe[bedpe["chr1"].astype(str).eq(args.chrom) & bedpe["chr2"].astype(str).eq(args.chrom)].copy()

    for col in ["re_start", "re_end", "prom_start", "prom_end"]:
        edges[col] = pd.to_numeric(edges[col], errors="coerce").fillna(0).astype(np.int64)
    edges["chr"] = edges["chr"].astype(str)

    labels = np.zeros(len(edges), dtype=np.int8)
    total_loops = 0
    for chrom, edges_chr in edges.groupby("chr", sort=False):
        bedpe_chr = bedpe[bedpe["chr1"].eq(chrom) & bedpe["chr2"].eq(chrom)].copy()
        total_loops += len(bedpe_chr)
        if bedpe_chr.empty:
            continue

        loop_ranges = []
        for loop_idx, loop in bedpe_chr.iterrows():
            left, right = _loop_bounds(loop)
            loop_ranges.append((left - args.anchor_slop, right + args.anchor_slop, loop_idx))

        for edge_idx, edge in edges_chr.iterrows():
            edge_left = min(int(edge["re_start"]), int(edge["prom_start"])) - args.anchor_slop
            edge_right = max(int(edge["re_end"]), int(edge["prom_end"])) + args.anchor_slop
            for loop_left, loop_right, loop_idx in loop_ranges:
                if loop_right < edge_left or edge_right < loop_left:
                    continue
                if _edge_hits_loop(edge, bedpe_chr.loc[loop_idx], args.anchor_slop):
                    labels[edges.index.get_loc(edge_idx)] = 1
                    break

    edges["edge_label"] = labels
    edges.to_csv(output_path, sep="\t", index=False)
    print(f"ABC edges labeled: {len(edges)}")
    print(f"Hi-C loops considered: {total_loops}")
    print(f"Positive edges: {int(labels.sum())}")
    print(f"Saved edge labels to {output_path}")


if __name__ == "__main__":
    main()
