from __future__ import annotations

from typing import Dict, Iterable, Set, Tuple

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


def load_edge_label_table(
    edge_label_path: str,
    node_table: pd.DataFrame,
    label_col: str,
    re_col: str = "re_node_id",
    promoter_col: str = "promoter_node_id",
) -> pd.DataFrame:
    edge_df = pd.read_csv(edge_label_path, sep="\t").copy()
    required = [re_col, promoter_col, label_col]
    missing = [col for col in required if col not in edge_df.columns]
    if missing:
        raise KeyError(f"Edge label table is missing required columns: {missing}")
    if "node_id" not in node_table.columns:
        raise KeyError("node_table must contain a node_id column")

    node_to_idx = {str(node_id): idx for idx, node_id in enumerate(node_table["node_id"].astype(str))}
    src_idx = edge_df[re_col].astype(str).map(node_to_idx)
    dst_idx = edge_df[promoter_col].astype(str).map(node_to_idx)
    keep = src_idx.notna() & dst_idx.notna()

    labels = pd.to_numeric(edge_df.loc[keep, label_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    table = pd.DataFrame(
        {
            "i": src_idx.loc[keep].astype(int).to_numpy(dtype=np.int64),
            "j": dst_idx.loc[keep].astype(int).to_numpy(dtype=np.int64),
            "label": (labels > 0.5).astype(np.int8),
        }
    )
    return table.reset_index(drop=True)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    pos_rank_sum = float(ranks[labels].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int8)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(len(sorted_labels), dtype=np.float64) + 1.0)
    return float(precision[sorted_labels == 1].sum() / n_pos)


def evaluate_edge_label_ranking(
    predicted_adj: Tensor,
    edge_label_table: pd.DataFrame,
    topks: Iterable[int] = (500, 1000, 2000),
) -> Dict[str, float]:
    if predicted_adj.is_sparse:
        predicted_adj = predicted_adj.to_dense()
    dense_scores = predicted_adj.detach().cpu().numpy()
    i = edge_label_table["i"].to_numpy(dtype=np.int64)
    j = edge_label_table["j"].to_numpy(dtype=np.int64)
    labels = edge_label_table["label"].to_numpy(dtype=np.int8)
    scores = dense_scores[i, j].astype(np.float64)

    valid = np.isfinite(scores)
    scores = scores[valid]
    labels = labels[valid]
    n_edges = int(len(labels))
    n_pos = int(labels.sum())
    prevalence = float(n_pos) / float(n_edges) if n_edges > 0 else 0.0

    metrics: Dict[str, float] = {
        "num_edges": float(n_edges),
        "positive_edges": float(n_pos),
        "prevalence": prevalence,
        "auroc": _roc_auc(labels, scores) if n_edges > 0 else float("nan"),
        "auprc": _average_precision(labels, scores) if n_edges > 0 else float("nan"),
    }
    if n_edges == 0:
        return metrics

    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_labels = labels[order]
    for requested_k in topks:
        k = int(requested_k)
        if k <= 0:
            continue
        actual_k = min(k, n_edges)
        hits = int(sorted_labels[:actual_k].sum())
        precision = float(hits) / float(actual_k) if actual_k > 0 else 0.0
        recall = float(hits) / float(n_pos) if n_pos > 0 else 0.0
        enrichment = precision / prevalence if prevalence > 0 else float("nan")
        metrics[f"hit_count_at_{k}"] = float(hits)
        metrics[f"precision_at_{k}"] = precision
        metrics[f"recall_at_{k}"] = recall
        metrics[f"enrichment_at_{k}"] = enrichment
    return metrics
