from __future__ import annotations

from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from data_utils import bedpe_to_anchor_pairs


def filter_bedpe_to_node_chroms(hic_bedpe: pd.DataFrame, node_table: pd.DataFrame) -> pd.DataFrame:
    if {"chr1", "start1", "end1", "chr2", "start2", "end2"}.issubset(hic_bedpe.columns):
        bedpe = hic_bedpe[["chr1", "start1", "end1", "chr2", "start2", "end2"]].copy()
    else:
        bedpe = bedpe_to_anchor_pairs(hic_bedpe)
    node_chroms = set(node_table["chr"].astype(str).unique())
    bedpe["chr1"] = bedpe["chr1"].astype(str)
    bedpe["chr2"] = bedpe["chr2"].astype(str)
    keep = bedpe["chr1"].isin(node_chroms) & bedpe["chr2"].isin(node_chroms)
    return bedpe.loc[keep].reset_index(drop=True)


def evaluate_with_hic(
    predicted_adj: Tensor,
    node_table: pd.DataFrame,
    hic_bedpe: pd.DataFrame,
    topk: int = 1000,
) -> Dict[str, float]:
    if predicted_adj.is_sparse:
        predicted_adj = predicted_adj.to_dense()
    scores = predicted_adj.detach().cpu().numpy()
    n = scores.shape[0]
    if n < 2 or topk <= 0:
        return {"topk": 0.0, "hit_count": 0.0, "hit_rate": 0.0}

    upper_i, upper_j = np.triu_indices(n, k=1)
    upper_scores = scores[upper_i, upper_j]
    valid = np.isfinite(upper_scores)
    upper_i = upper_i[valid]
    upper_j = upper_j[valid]
    upper_scores = upper_scores[valid]
    if upper_scores.size == 0:
        return {"topk": 0.0, "hit_count": 0.0, "hit_rate": 0.0}

    actual_topk = min(topk, upper_scores.size)
    top_idx = np.argpartition(upper_scores, -actual_topk)[-actual_topk:]
    top_idx = top_idx[np.argsort(upper_scores[top_idx])[::-1]]
    pred_pairs = np.column_stack([upper_i[top_idx], upper_j[top_idx]])

    pred_table = pd.DataFrame(pred_pairs, columns=["i", "j"])
    pred_table["score"] = scores[pred_table["i"], pred_table["j"]]

    node_chr = node_table["chr"].astype(str).to_numpy()
    node_start = pd.to_numeric(node_table["start"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    node_end = pd.to_numeric(node_table["end"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)

    def _overlap(chr_a, s_a, e_a, chr_b, s_b, e_b) -> bool:
        if chr_a != chr_b:
            return False
        return not (e_a < s_b or e_b < s_a)

    bedpe = filter_bedpe_to_node_chroms(hic_bedpe, node_table)
    hits = 0
    for _, row in pred_table.iterrows():
        i = int(row["i"])
        j = int(row["j"])
        ok = False
        for _, loop in bedpe.iterrows():
            left = _overlap(node_chr[i], node_start[i], node_end[i], str(loop["chr1"]), int(loop["start1"]), int(loop["end1"]))
            right = _overlap(node_chr[j], node_start[j], node_end[j], str(loop["chr2"]), int(loop["start2"]), int(loop["end2"]))
            swapped_left = _overlap(node_chr[i], node_start[i], node_end[i], str(loop["chr2"]), int(loop["start2"]), int(loop["end2"]))
            swapped_right = _overlap(node_chr[j], node_start[j], node_end[j], str(loop["chr1"]), int(loop["start1"]), int(loop["end1"]))
            if (left and right) or (swapped_left and swapped_right):
                ok = True
                break
        hits += int(ok)

    return {
        "topk": float(actual_topk),
        "hit_count": float(hits),
        "hit_rate": float(hits) / float(actual_topk) if actual_topk > 0 else 0.0,
    }


def topk_edge_set(predicted_adj: Tensor, topk: int = 1000) -> Set[Tuple[int, int]]:
    if predicted_adj.is_sparse:
        predicted_adj = predicted_adj.to_dense()
    scores = predicted_adj.detach().cpu().numpy()
    n = scores.shape[0]
    if n < 2 or topk <= 0:
        return set()

    upper_i, upper_j = np.triu_indices(n, k=1)
    upper_scores = scores[upper_i, upper_j]
    valid = np.isfinite(upper_scores)
    upper_i = upper_i[valid]
    upper_j = upper_j[valid]
    upper_scores = upper_scores[valid]
    if upper_scores.size == 0:
        return set()

    actual_topk = min(topk, upper_scores.size)
    top_idx = np.argpartition(upper_scores, -actual_topk)[-actual_topk:]
    return {(int(upper_i[idx]), int(upper_j[idx])) for idx in top_idx}


def compare_topk_edges(init_adj: Tensor, optimized_adj: Tensor, topk: int = 1000) -> Dict[str, float]:
    init_edges = topk_edge_set(init_adj, topk=topk)
    opt_edges = topk_edge_set(optimized_adj, topk=topk)
    if not init_edges and not opt_edges:
        return {"topk_overlap": 0.0, "topk_jaccard": 0.0, "init_topk_size": 0.0, "opt_topk_size": 0.0}
    overlap = len(init_edges & opt_edges)
    union = len(init_edges | opt_edges)
    denom = min(len(init_edges), len(opt_edges))
    return {
        "topk_overlap": float(overlap) / float(denom) if denom > 0 else 0.0,
        "topk_jaccard": float(overlap) / float(union) if union > 0 else 0.0,
        "init_topk_size": float(len(init_edges)),
        "opt_topk_size": float(len(opt_edges)),
    }
