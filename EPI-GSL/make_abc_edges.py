from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


REQUIRED_NODE_COLS = ["node_id", "chr", "start", "end", "atac_signal_sum"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lightweight ABC RE-promoter edge table.")
    parser.add_argument("--promoter-path", type=str, default="promoter_nodes_full.tsv")
    parser.add_argument("--re-path", type=str, default="re_nodes_full.tsv")
    parser.add_argument("--output-path", type=str, default="work/abc_edges.tsv")
    parser.add_argument("--min-distance", type=int, default=2000)
    parser.add_argument("--max-distance", type=int, default=1_000_000)
    parser.add_argument("--chrom", type=str, default="", help="Optional chromosome subset, for example chr1.")
    parser.add_argument(
        "--activity-transform",
        choices=["raw", "log1p"],
        default="log1p",
        help="Transform for RE atac_signal_sum before computing ABC raw score.",
    )
    parser.add_argument(
        "--contact-power",
        type=float,
        default=-1.0,
        help="Power-law contact exponent. -1 means 1 / (distance + 1).",
    )
    parser.add_argument(
        "--min-abc-score",
        type=float,
        default=0.0,
        help="Drop edges with abc_score <= this value after per-promoter normalization.",
    )
    return parser.parse_args()


def _check_cols(df: pd.DataFrame, name: str) -> None:
    missing = [col for col in REQUIRED_NODE_COLS if col not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def _prepare_nodes(path: str, source: str, chrom: str = "") -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t").copy()
    _check_cols(df, path)
    if chrom:
        df = df[df["chr"].astype(str).eq(chrom)].copy()
    df["node_source"] = source
    df["start"] = pd.to_numeric(df["start"], errors="coerce").fillna(0).astype(np.int64)
    df["end"] = pd.to_numeric(df["end"], errors="coerce").fillna(0).astype(np.int64)
    df["center"] = ((df["start"] + df["end"]) / 2.0).astype(np.float64)
    df["atac_signal_sum"] = pd.to_numeric(df["atac_signal_sum"], errors="coerce").fillna(0.0)
    return df.reset_index(drop=True)


def _activity(values: np.ndarray, transform: str) -> np.ndarray:
    values = np.clip(values.astype(np.float64), a_min=0.0, a_max=None)
    if transform == "log1p":
        return np.log1p(values)
    return values


def _write_chunk(path: Path, rows: List[dict], write_header: bool) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, mode="a", header=write_header)


def _chrom_order(promoters: pd.DataFrame, res: pd.DataFrame) -> Iterable[str]:
    chroms = sorted(set(promoters["chr"].astype(str)).intersection(set(res["chr"].astype(str))))
    return chroms


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    promoters = _prepare_nodes(args.promoter_path, "promoter", args.chrom)
    res = _prepare_nodes(args.re_path, "re", args.chrom)

    print(f"Promoters={len(promoters)} REs={len(res)}")
    print(
        f"Distance window=[{args.min_distance}, {args.max_distance}] "
        f"activity={args.activity_transform} contact_power={args.contact_power}"
    )

    total_edges = 0
    write_header = True
    for chrom in _chrom_order(promoters, res):
        p_chr = promoters[promoters["chr"].astype(str).eq(chrom)].copy()
        re_chr = res[res["chr"].astype(str).eq(chrom)].copy().sort_values("center").reset_index(drop=True)
        re_centers = re_chr["center"].to_numpy(dtype=np.float64)
        re_activity = _activity(re_chr["atac_signal_sum"].to_numpy(dtype=np.float64), args.activity_transform)

        rows: List[dict] = []
        for _, prom in p_chr.iterrows():
            prom_center = float(prom["center"])
            left = np.searchsorted(re_centers, prom_center - args.max_distance, side="left")
            right = np.searchsorted(re_centers, prom_center + args.max_distance, side="right")
            if left >= right:
                continue

            cand = re_chr.iloc[left:right].copy()
            distances = np.abs(re_centers[left:right] - prom_center)
            keep = (distances >= args.min_distance) & (distances <= args.max_distance)
            if not np.any(keep):
                continue

            cand = cand.loc[keep].copy()
            kept_distances = distances[keep]
            activity = re_activity[left:right][keep]
            contact = np.power(kept_distances + 1.0, args.contact_power)
            raw = activity * contact
            denom = raw.sum()
            if denom <= 0:
                continue
            abc_score = raw / denom

            score_keep = abc_score > args.min_abc_score
            if not np.any(score_keep):
                continue

            cand = cand.loc[score_keep]
            kept_distances = kept_distances[score_keep]
            activity = activity[score_keep]
            contact = contact[score_keep]
            raw = raw[score_keep]
            abc_score = abc_score[score_keep]

            for re_row, dist, act, cont, raw_score, score in zip(
                cand.itertuples(index=False),
                kept_distances,
                activity,
                contact,
                raw,
                abc_score,
            ):
                rows.append(
                    {
                        "re_node_id": re_row.node_id,
                        "promoter_node_id": prom["node_id"],
                        "gene_name": prom.get("gene_name", ""),
                        "chr": chrom,
                        "re_start": int(re_row.start),
                        "re_end": int(re_row.end),
                        "prom_start": int(prom["start"]),
                        "prom_end": int(prom["end"]),
                        "distance": int(round(float(dist))),
                        "activity": float(act),
                        "contact": float(cont),
                        "abc_raw": float(raw_score),
                        "abc_score": float(score),
                    }
                )

        _write_chunk(output_path, rows, write_header=write_header)
        write_header = False
        total_edges += len(rows)
        print(f"{chrom}: wrote {len(rows)} ABC edges")

    print(f"Saved {total_edges} ABC edges to {output_path}")


if __name__ == "__main__":
    main()
