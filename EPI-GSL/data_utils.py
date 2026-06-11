from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor


DEFAULT_META_COLS = [
    "node_id",
    "node_type",
    "chr",
    "start",
    "end",
    "gene_name",
    "strand",
    "atac_peak_count",
    "atac_signal_sum",
    "atac_signal_max",
    "atac_max_overlap",
]


def _safe_zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    mean = values.mean()
    std = values.std()
    if std == 0:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def _collect_feature_columns(df: pd.DataFrame, preferred_meta: Sequence[str]) -> List[str]:
    meta = set(preferred_meta)
    numeric_cols = []
    for col in df.columns:
        if col in meta:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            numeric_cols.append(col)
    return numeric_cols


def _align_feature_columns(
    promoter_df: pd.DataFrame,
    re_df: pd.DataFrame,
    preferred_meta: Sequence[str] = DEFAULT_META_COLS,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    promoter_features = _collect_feature_columns(promoter_df, preferred_meta)
    re_features = _collect_feature_columns(re_df, preferred_meta)
    feature_cols = sorted(set(promoter_features).union(set(re_features)))

    for col in feature_cols:
        if col not in promoter_df.columns:
            promoter_df[col] = 0.0
        if col not in re_df.columns:
            re_df[col] = 0.0

    promoter_df[feature_cols] = promoter_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    re_df[feature_cols] = re_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return promoter_df, re_df, feature_cols


def load_peak_node_tables(
    promoter_path: str,
    re_path: str,
    feature_cols: Optional[Sequence[str]] = None,
    label_col: str = "atac_signal_sum",
    zscore_labels: bool = True,
) -> Tuple[pd.DataFrame, Tensor, Tensor, List[str]]:
    promoter_df = pd.read_csv(promoter_path, sep="\t").copy()
    re_df = pd.read_csv(re_path, sep="\t").copy()

    promoter_df["node_source"] = "promoter"
    re_df["node_source"] = "re"


    promoter_df, re_df, inferred_features = _align_feature_columns(promoter_df, re_df)
    if feature_cols is None:
        feature_cols = inferred_features
    else:
        feature_cols = list(feature_cols)
        for col in feature_cols:
            if col not in promoter_df.columns:
                promoter_df[col] = 0.0
            if col not in re_df.columns:
                re_df[col] = 0.0
        promoter_df[feature_cols] = promoter_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        re_df[feature_cols] = re_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    node_table = pd.concat([promoter_df, re_df], ignore_index=True, sort=False).reset_index(drop=True)

    if label_col not in node_table.columns:
        raise KeyError(f"Missing label column: {label_col}")

    labels = pd.to_numeric(node_table[label_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    if zscore_labels:
        labels = _safe_zscore(labels)

    features = node_table[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return node_table, torch.from_numpy(features), torch.from_numpy(labels), list(feature_cols)


def build_candidate_adj_from_distance(
    node_table: pd.DataFrame,
    max_distance: Optional[int] = None,
    same_chrom_only: bool = True,
    ep_only: bool = True,
    symmetric: bool = True,
    normalize: bool = True,
) -> Tensor:
    def _is_promoter(row: pd.Series) -> bool:
        src = str(row.get("node_source", "")).lower()
        if src:
            return src in {"promoter", "prom"}
        t = str(row.get("node_type", "")).lower()
        return t in {"gene", "promoter", "prom"}

    def _is_enhancer(row: pd.Series) -> bool:
        src = str(row.get("node_source", "")).lower()
        if src:
            return src in {"re", "enhancer", "ccre", "distal"}
        t = str(row.get("node_type", "")).lower()
        return t in {"re", "enhancer", "ccre", "distal"}

    coords_chr = node_table["chr"].astype(str).to_numpy()
    starts = pd.to_numeric(node_table["start"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    ends = pd.to_numeric(node_table["end"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    centers = ((starts + ends) / 2.0).astype(np.float64)
    is_prom = node_table.apply(_is_promoter, axis=1).to_numpy()
    is_enh = node_table.apply(_is_enhancer, axis=1).to_numpy()

    n = len(node_table)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if ep_only:
                cross_type = (is_enh[i] and is_prom[j]) or (is_prom[i] and is_enh[j])
                if not cross_type:
                    continue
            if same_chrom_only and coords_chr[i] != coords_chr[j]:
                continue
            dist = abs(centers[i] - centers[j])
            if max_distance is not None and dist > max_distance:
                continue
            weight = 1.0 / (1.0 + dist) if normalize else 1.0
            adj[i, j] = weight
            if symmetric:
                adj[j, i] = weight
    np.fill_diagonal(adj, 1.0)
    return torch.from_numpy(adj)


def load_abc_edges(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def build_candidate_adj_from_abc_edges(
    node_table: pd.DataFrame,
    abc_edges: pd.DataFrame,
    score_col: str = "abc_score",
    re_col: str = "re_node_id",
    promoter_col: str = "promoter_node_id",
    symmetric: bool = True,
    include_self: bool = True,
) -> Tensor:
    missing = [col for col in [re_col, promoter_col, score_col] if col not in abc_edges.columns]
    if missing:
        raise KeyError(f"Missing columns in ABC edge table: {missing}")
    if "node_id" not in node_table.columns:
        raise KeyError("node_table must contain a node_id column")

    node_to_idx = {str(node_id): idx for idx, node_id in enumerate(node_table["node_id"].astype(str))}
    n = len(node_table)
    adj = np.zeros((n, n), dtype=np.float32)

    edge_re = abc_edges[re_col].astype(str).to_numpy()
    edge_promoter = abc_edges[promoter_col].astype(str).to_numpy()
    edge_score = pd.to_numeric(abc_edges[score_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    kept = 0
    for re_id, promoter_id, score in zip(edge_re, edge_promoter, edge_score):
        if score <= 0:
            continue
        re_idx = node_to_idx.get(re_id)
        promoter_idx = node_to_idx.get(promoter_id)
        if re_idx is None or promoter_idx is None:
            continue
        if score > adj[re_idx, promoter_idx]:
            adj[re_idx, promoter_idx] = score
            if symmetric:
                adj[promoter_idx, re_idx] = score
        kept += 1

    if include_self:
        np.fill_diagonal(adj, 1.0)
    print(f"Loaded ABC adjacency edges kept={kept} nodes={n}")
    return torch.from_numpy(adj)


def load_hic_bedpe(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", comment="#", header=None)


def bedpe_to_anchor_pairs(
    bedpe: pd.DataFrame,
    chr1_col: int = 0,
    start1_col: int = 1,
    end1_col: int = 2,
    chr2_col: int = 3,
    start2_col: int = 4,
    end2_col: int = 5,
) -> pd.DataFrame:
    anchors = bedpe[[chr1_col, start1_col, end1_col, chr2_col, start2_col, end2_col]].copy()
    anchors.columns = ["chr1", "start1", "end1", "chr2", "start2", "end2"]
    return anchors
