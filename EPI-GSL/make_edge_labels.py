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
    parser.add_argument("--test-fraction", type=float, default=0.0, help="Fraction of Hi-C loops held out for test labels.")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for train/test Hi-C loop split.")
    parser.add_argument("--train-bedpe-output", type=str, default="", help="Optional output BEDPE for train loops.")
    parser.add_argument("--test-bedpe-output", type=str, default="", help="Optional output BEDPE for held-out test loops.")
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


def _label_edges(edges: pd.DataFrame, bedpe: pd.DataFrame, slop: int) -> Tuple[np.ndarray, int]:
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
            loop_ranges.append((left - slop, right + slop, loop_idx))

        for edge_idx, edge in edges_chr.iterrows():
            edge_left = min(int(edge["re_start"]), int(edge["prom_start"])) - slop
            edge_right = max(int(edge["re_end"]), int(edge["prom_end"])) + slop
            for loop_left, loop_right, loop_idx in loop_ranges:
                if loop_right < edge_left or edge_right < loop_left:
                    continue
                if _edge_hits_loop(edge, bedpe_chr.loc[loop_idx], slop):
                    labels[edges.index.get_loc(edge_idx)] = 1
                    break
    return labels, total_loops


def _split_bedpe(bedpe: pd.DataFrame, test_fraction: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if test_fraction <= 0:
        return bedpe.reset_index(drop=True), bedpe.iloc[0:0].copy().reset_index(drop=True)
    if test_fraction >= 1:
        raise ValueError("--test-fraction must be < 1.0")
    rng = np.random.default_rng(seed)
    is_test = rng.random(len(bedpe)) < test_fraction
    if is_test.all() and len(is_test) > 0:
        is_test[rng.integers(0, len(is_test))] = False
    if (not is_test.any()) and len(is_test) > 1:
        is_test[rng.integers(0, len(is_test))] = True
    train_bedpe = bedpe.loc[~is_test].reset_index(drop=True)
    test_bedpe = bedpe.loc[is_test].reset_index(drop=True)
    return train_bedpe, test_bedpe


def _write_bedpe(path: str, bedpe: pd.DataFrame) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bedpe[["chr1", "start1", "end1", "chr2", "start2", "end2"]].to_csv(
        output_path,
        sep="\t",
        index=False,
        header=False,
    )


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

    train_bedpe, test_bedpe = _split_bedpe(bedpe, args.test_fraction, args.split_seed)
    train_labels, train_loops = _label_edges(edges, train_bedpe, args.anchor_slop)
    edges["edge_label"] = train_labels
    if args.test_fraction > 0:
        test_labels, test_loops = _label_edges(edges, test_bedpe, args.anchor_slop)
        edges["edge_label_train"] = train_labels
        edges["edge_label_test"] = test_labels
    else:
        test_labels = np.zeros(len(edges), dtype=np.int8)
        test_loops = 0

    edges.to_csv(output_path, sep="\t", index=False)
    _write_bedpe(args.train_bedpe_output, train_bedpe)
    _write_bedpe(args.test_bedpe_output, test_bedpe)

    print(f"ABC edges labeled: {len(edges)}")
    print(f"Hi-C loops total after chrom filter: {len(bedpe)}")
    print(f"Hi-C train loops: {train_loops}")
    print(f"Train positive edges: {int(train_labels.sum())}")
    if args.test_fraction > 0:
        print(f"Hi-C test loops: {test_loops}")
        print(f"Test positive edges: {int(test_labels.sum())}")
        if args.train_bedpe_output:
            print(f"Saved train BEDPE to {args.train_bedpe_output}")
        if args.test_bedpe_output:
            print(f"Saved test BEDPE to {args.test_bedpe_output}")
    print(f"Saved edge labels to {output_path}")


if __name__ == "__main__":
    main()
