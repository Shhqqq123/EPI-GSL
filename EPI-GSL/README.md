# EPI-GSL

Modularized peak-level IDGL prototype for enhancer-promoter graph optimization.

## Modules
- `data_utils.py`: node table loading, candidate adjacency construction, Hi-C bedpe loading
- `graph_utils.py`: dense adjacency helpers, encoders, graph learner, refiner
- `model.py`: `PeakLevelIDGLPyG`
- `loss.py`: `PeakLevelIDGLLoss`
- `eval_utils.py`: Hi-C top-k evaluation
- `train.py`: minimal training entry
- `eval.py`: compare initial vs optimized graph against Hi-C

## Examples
```powershell
python EPI-GSL\train.py --promoter-path promoter_nodes_full.tsv --re-path re_nodes_full.tsv --ep-only
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

Train with ABC scores as the initial adjacency:

```powershell
python EPI-GSL\train.py `
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
