# EPI-GSL

Peak-level enhancer-promoter graph learning and ABC candidate edge reranking.

The current recommended path is `--model-mode edge-rerank`: it avoids materializing a full
`N x N` adjacency matrix and trains directly on ABC candidate edges. The model learns a
residual correction on top of ABC:

```text
final_score = logit(ABC_score) + model_delta(motif/node features, edge features)
```

This is better suited for tens of thousands of promoter/enhancer peaks than the original
dense IDGL prototype. The default reranker is conservative: `model_delta` is scaled before
being added to the ABC logit, and delta regularization keeps the final ranking close to ABC.

## Modules
- `data_utils.py`: node table loading, candidate adjacency construction, Hi-C bedpe loading
- `graph_utils.py`: dense adjacency helpers, encoders, graph learner, refiner
- `model.py`: `PeakLevelIDGLPyG` and `EdgeResidualReranker`
- `loss.py`: `PeakLevelIDGLLoss`
- `eval_utils.py`: Hi-C top-k evaluation
- `train.py`: training entry for dense IDGL or scalable residual edge reranking
- `eval.py`: compare ABC vs residual edge-rerank scores against Hi-C or edge labels

## Examples
```powershell
python EPI-GSL\train.py --model-mode dense-idgl --promoter-path promoter_nodes_full.tsv --re-path re_nodes_full.tsv --ep-only
python EPI-GSL\eval.py --outputs-path outputs\epi_gsl\ep_idgl_outputs.pt --ep-only
```

## Lightweight ABC Initial Graph

Build RE-promoter ABC edges from the existing node tables:

```powershell
python EPI-GSL\make_abc_edges.py `
  --promoter-path promoter_nodes_full.tsv `
  --re-path re_nodes_full.tsv `
  --output-path work\abc_edges_chr1.tsv `
  --chrom chr1 `
  --min-distance 2000 `
  --max-distance 1000000 `
  --activity-transform log1p `
  --contact-power -1
```

Train the scalable residual edge reranker:

```powershell
python EPI-GSL\train.py `
  --model-mode edge-rerank `
  --promoter-path promoter_nodes_full.tsv `
  --re-path re_nodes_full.tsv `
  --output-dir outputs\epi_gsl_edge_rerank_chr5 `
  --edge-labels work\edge_labels_chr5_split.tsv `
  --edge-label-col edge_label_train `
  --abc-edges work\abc_edges_chr5.tsv `
  --chrom chr5 `
  --sample-size 5000 `
  --epochs 100 `
  --hidden-dim 128 `
  --negative-sampling abc-distance-matched `
  --negative-ratio 5 `
  --ranking-loss-weight 0.1 `
  --ranking-negatives-per-positive 2 `
  --delta-l2-weight 0.001 `
  --delta-logit-scale 0.25 `
  --device cpu
```

Evaluate the residual edge scores on held-out labels:

```powershell
python EPI-GSL\eval.py `
  --outputs-path outputs\epi_gsl_edge_rerank_chr5\ep_idgl_outputs.pt `
  --hic-bedpe work\hic_chr5_test.bedpe `
  --edge-labels work\edge_labels_chr5_split.tsv `
  --edge-label-col edge_label_test `
  --topk 1000 `
  --edge-metric-topks 500,1000,2000 `
  --metrics-output outputs\epi_gsl_edge_rerank_chr5\eval_metrics.json
```

The main output for downstream analysis is:

```text
outputs\epi_gsl_edge_rerank_chr5\edge_scores.tsv
```

It contains each candidate edge with `abc_score`, `edge_delta_logit`, `edge_logit`,
and `final_score`.

Sparse iterative graph structure learning mode:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File EPI-GSL\run_edge_seed_splits.ps1 `
  -ModelMode sparse-gsl `
  -Seeds "1,2,3,4,5" `
  -SampleSize 10000 `
  -GraphIters 2 `
  -DeltaLogitScale 0.1 `
  -DeltaL2Weight 0.001 `
  -OutputRoot outputs\sparse_gsl_seed_splits_chr5
```

Sparse GSL with ABC hard-negative top-rank training:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File EPI-GSL\run_edge_seed_splits.ps1 `
  -ModelMode sparse-gsl `
  -Seeds "1,2,3,4,5" `
  -SampleSize 10000 `
  -GraphIters 2 `
  -DeltaLogitScale 0.075 `
  -DeltaL2Weight 0.003 `
  -RankingLossWeight 0.1 `
  -HardNegativeRatio 2 `
  -HardRankLossWeight 0.2 `
  -HardRankMargin 0.5 `
  -HardRankNegativesPerPositive 4 `
  -HardRankTopNegativeRatio 20 `
  -OutputRoot outputs\sparse_gsl_hardrank_g2_scale0p075_l2p003
```

Sparse GSL with ABC rank consistency:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File EPI-GSL\run_loco_multi_chrom.ps1 `
  -TestChroms "chr1,chr5,chr10,chr15,chr20" `
  -ModelMode sparse-gsl `
  -SampleSize 10000 `
  -GraphIters 2 `
  -DeltaLogitScale 0.075 `
  -DeltaL2Weight 0.003 `
  -RankingLossWeight 0.1 `
  -HardNegativeRatio 2 `
  -HardRankLossWeight 0.1 `
  -AbcRankLossWeight 0.05 `
  -AbcRankMargin 0.02 `
  -AbcRankMinScoreGap 0.0001 `
  -AbcRankScope negatives `
  -OutputRoot outputs\loco_multi_sparse_gsl_hardrank_abcrank
```

Sparse GSL with validation early stopping:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File EPI-GSL\run_loco_multi_chrom.ps1 `
  -TestChroms "chr1,chr5,chr10,chr15,chr20" `
  -ModelMode sparse-gsl `
  -SampleSize 10000 `
  -GraphIters 2 `
  -DeltaLogitScale 0.075 `
  -DeltaL2Weight 0.003 `
  -RankingLossWeight 0.1 `
  -HardNegativeRatio 2 `
  -HardRankLossWeight 0.1 `
  -AbcRankLossWeight 0.02 `
  -AbcRankMargin 0.02 `
  -AbcRankMinScoreGap 0.0001 `
  -AbcRankScope negatives `
  -ValidationFraction 0.1 `
  -ValidationMetric auprc `
  -EarlyStoppingPatience 30 `
  -EarlyStoppingMinEpochs 80 `
  -ScoreBlendAlphas "0,0.05,0.1,0.2,0.3,0.5,0.7,1" `
  -ScoreBlendMethod rank `
  -ScoreBlendMetric auprc `
  -OutputRoot outputs\loco_multi_sparse_gsl_hardrank_abcrank_blend
```

When `ScoreBlendAlphas` is provided, training selects the alpha with the best validation
AUPRC and prediction writes `blended_score`; evaluation uses `blended_score` automatically
when it is available.

For chromosome-level validation instead of random edge validation, add
`-AutoValidationChrom`. Each held-out run uses the next available chromosome after the
test chromosome for validation and excludes those validation-chromosome edges from the
training loss. Use `-AutoValidationChromStrategy previous` or
`-AutoValidationChromCount 2` for alternative validation splits.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File EPI-GSL\run_loco_multi_chrom.ps1 `
  -TestChroms "chr1,chr5,chr10,chr15,chr20" `
  -ModelMode sparse-gsl `
  -SampleSize 10000 `
  -GraphIters 2 `
  -DeltaLogitScale 0.075 `
  -DeltaL2Weight 0.003 `
  -RankingLossWeight 0.1 `
  -HardNegativeRatio 2 `
  -HardRankLossWeight 0.1 `
  -AbcRankLossWeight 0.02 `
  -AbcRankMargin 0.02 `
  -AbcRankMinScoreGap 0.0001 `
  -AbcRankScope negatives `
  -AutoValidationChrom `
  -AutoValidationChromStrategy next `
  -AutoValidationChromCount 1 `
  -ValidationMetric auprc `
  -EarlyStoppingPatience 30 `
  -EarlyStoppingMinEpochs 80 `
  -ScoreBlendAlphas "0,0.05,0.1,0.2,0.3,0.5,0.7,1" `
  -ScoreBlendMethod rank `
  -ScoreBlendMetric auprc `
  -OutputRoot outputs\loco_multi_sparse_gsl_hardrank_abcrank_chromval_blend
```

For chromosome-batch training, add `-ChromBatchTraining` and set `-SampleSize 0`.
This trains one chromosome graph at a time, keeping each chromosome's ABC candidate
graph intact instead of random-sampling nodes across chromosomes:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File EPI-GSL\run_loco_multi_chrom.ps1 `
  -TestChroms "chr1,chr5,chr10,chr15,chr20" `
  -ModelMode sparse-gsl `
  -ChromBatchTraining `
  -SampleSize 0 `
  -GraphIters 2 `
  -DeltaLogitScale 0.075 `
  -DeltaL2Weight 0.003 `
  -RankingLossWeight 0.1 `
  -HardNegativeRatio 2 `
  -HardRankLossWeight 0.1 `
  -AbcRankLossWeight 0.02 `
  -AbcRankMargin 0.02 `
  -AbcRankMinScoreGap 0.0001 `
  -AbcRankScope negatives `
  -AutoValidationChrom `
  -AutoValidationChromStrategy next `
  -AutoValidationChromCount 1 `
  -ValidationMetric auprc `
  -EarlyStoppingPatience 30 `
  -EarlyStoppingMinEpochs 80 `
  -ScoreBlendAlphas "0,0.05,0.1,0.2,0.3,0.5,0.7,1" `
  -ScoreBlendMethod rank `
  -ScoreBlendMetric auprc `
  -OutputRoot outputs\loco_multi_sparse_gsl_chrombatch_fullgraph
```

LOCO with sparse iterative graph learning:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File EPI-GSL\run_loco_chrom_split.ps1 `
  -ModelMode sparse-gsl `
  -SampleSize 10000 `
  -GraphIters 2 `
  -DeltaLogitScale 0.1 `
  -DeltaL2Weight 0.001 `
  -OutputRoot outputs\loco_chr5_sparse_gsl_scale0p1_l2p001_sample10k
```

Run a conservative residual parameter grid:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File EPI-GSL\run_edge_rerank_grid.ps1 `
  -Seeds "1,2,3,4,5" `
  -DeltaLogitScales "0.1,0.25,0.5" `
  -DeltaL2Weights "0.001,0.01" `
  -GridRoot outputs\edge_rerank_param_grid_chr5
```

The final comparison table is:

```text
outputs\edge_rerank_param_grid_chr5\grid_summary.tsv
```

Original dense graph-learning mode:

```powershell
python EPI-GSL\train.py `
  --model-mode dense-idgl `
  --promoter-path promoter_nodes_full.tsv `
  --re-path re_nodes_full.tsv `
  --output-dir outputs\epi_gsl_abc_chr1 `
  --abc-edges work\abc_edges_chr1.tsv `
  --chrom chr1 `
  --sample-size 5000 `
  --epochs 100 `
  --graph-alpha 0.7 `
  --topk-edges 30 `
  --graph-iters 3 `
  --stability-weight 0.01 `
  --ep-only
```

Length-normalized variant for reducing promoter/cCRE length bias:

```powershell
python EPI-GSL\train.py `
  --model-mode dense-idgl `
  --promoter-path promoter_nodes_full.tsv `
  --re-path re_nodes_full.tsv `
  --output-dir outputs\epi_gsl_abc_chr1_iter3_len_norm `
  --abc-edges work\abc_edges_chr1.tsv `
  --chrom chr1 `
  --sample-size 5000 `
  --label-col log1p_atac_signal_per_kb `
  --normalize-features-by-length `
  --epochs 300 `
  --lr 0.0005 `
  --graph-alpha 0.5 `
  --topk-edges 50 `
  --graph-iters 3 `
  --stability-weight 0.001 `
  --smooth-weight 0.01 `
  --ep-only
```

Evaluate against Hi-C loops using the same ABC initial graph:

```powershell
python EPI-GSL\eval.py `
  --outputs-path outputs\epi_gsl_abc_chr1\ep_idgl_outputs.pt `
  --abc-edges work\abc_edges_chr1.tsv `
  --hic-bedpe ENCFF308MMM.bedpe\ENCFF308MMM.bedpe `
  --topk 1000 `
  --ep-only
```
