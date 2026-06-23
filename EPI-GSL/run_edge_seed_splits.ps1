param(
    [string]$PythonExe = "python",
    [string]$ProjectRoot = "D:\Documents\Playground",
    [string]$PromoterPath = "D:\Documents\Playground\promoter_nodes_full.tsv",
    [string]$RePath = "D:\Documents\Playground\re_nodes_full.tsv",
    [string]$HicBedpe = "D:\Documents\Playground\ENCFF308MMM.bedpe\ENCFF308MMM.bedpe",
    [string]$Chrom = "chr5",
    [string]$Seeds = "1,2,3,4,5",
    [double]$TestFraction = 0.2,
    [int]$SampleSize = 5000,
    [int]$Epochs = 300,
    [double]$Lr = 0.0005,
    [double]$GraphAlpha = 0.5,
    [int]$TopKEdges = 50,
    [int]$GraphIters = 3,
    [double]$StabilityWeight = 0.001,
    [double]$SmoothWeight = 0.01,
    [double]$EdgeLossWeight = 1.0,
    [double]$NegativeRatio = 5.0,
    [string]$LabelCol = "log1p_atac_signal_per_kb",
    [string]$EvalTopK = "1000",
    [string]$EdgeMetricTopKs = "500,1000,2000",
    [string]$WorkDir = "D:\Documents\Playground\work",
    [string]$OutputRoot = "D:\Documents\Playground\outputs\edge_seed_splits_chr5",
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Parse-IntList {
    param([string]$Text)
    if ([string]::IsNullOrWhiteSpace($Text)) { return @() }
    return @($Text -split "[,\s;]+" | Where-Object { $_ -ne "" } | ForEach-Object { [int]$_ })
}

function Invoke-Step {
    param(
        [string]$Name,
        [string]$LogPath,
        [string[]]$CommandArgs
    )
    Write-Host ""
    Write-Host "[$Name] $PythonExe $($CommandArgs -join ' ')"
    if ($DryRun) {
        return
    }
    $output = & $PythonExe @CommandArgs 2>&1
    $exitCode = $LASTEXITCODE
    $output | Tee-Object -FilePath $LogPath | Out-Host
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode"
    }
}

$makeLabelsScript = Join-Path $ProjectRoot "EPI-GSL\make_edge_labels.py"
$trainScript = Join-Path $ProjectRoot "EPI-GSL\train.py"
$evalScript = Join-Path $ProjectRoot "EPI-GSL\eval.py"
$summaryScript = Join-Path $ProjectRoot "EPI-GSL\summarize_seed_metrics.py"
$abcEdges = Join-Path $WorkDir ("abc_edges_{0}.tsv" -f $Chrom)

foreach ($path in @($makeLabelsScript, $trainScript, $evalScript, $summaryScript, $abcEdges)) {
    if (-not (Test-Path $path)) {
        throw "Required path not found: $path"
    }
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
Set-Location $ProjectRoot

$seedList = Parse-IntList $Seeds
foreach ($seed in $seedList) {
    $seedTag = "seed$seed"
    $edgeLabels = Join-Path $WorkDir ("edge_labels_{0}_{1}.tsv" -f $Chrom, $seedTag)
    $trainBedpe = Join-Path $WorkDir ("hic_{0}_train_{1}.bedpe" -f $Chrom, $seedTag)
    $testBedpe = Join-Path $WorkDir ("hic_{0}_test_{1}.bedpe" -f $Chrom, $seedTag)
    $runDir = Join-Path $OutputRoot $seedTag
    $metricsJson = Join-Path $runDir "eval_test_metrics.json"
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null

    Invoke-Step -Name "make-labels $seedTag" -LogPath (Join-Path $runDir "make_edge_labels.log") -CommandArgs @(
        $makeLabelsScript,
        "--abc-edges", $abcEdges,
        "--hic-bedpe", $HicBedpe,
        "--output-path", $edgeLabels,
        "--chrom", $Chrom,
        "--test-fraction", "$TestFraction",
        "--split-seed", "$seed",
        "--train-bedpe-output", $trainBedpe,
        "--test-bedpe-output", $testBedpe
    )

    Invoke-Step -Name "train $seedTag" -LogPath (Join-Path $runDir "train.log") -CommandArgs @(
        $trainScript,
        "--promoter-path", $PromoterPath,
        "--re-path", $RePath,
        "--output-dir", $runDir,
        "--abc-edges", $abcEdges,
        "--edge-labels", $edgeLabels,
        "--edge-label-col", "edge_label_train",
        "--chrom", $Chrom,
        "--sample-size", "$SampleSize",
        "--label-col", $LabelCol,
        "--normalize-features-by-length",
        "--epochs", "$Epochs",
        "--lr", "$Lr",
        "--graph-alpha", "$GraphAlpha",
        "--topk-edges", "$TopKEdges",
        "--graph-iters", "$GraphIters",
        "--stability-weight", "$StabilityWeight",
        "--smooth-weight", "$SmoothWeight",
        "--edge-loss-weight", "$EdgeLossWeight",
        "--negative-ratio", "$NegativeRatio",
        "--seed", "$seed",
        "--ep-only"
    )

    Invoke-Step -Name "eval $seedTag" -LogPath (Join-Path $runDir "eval_test.log") -CommandArgs @(
        $evalScript,
        "--outputs-path", (Join-Path $runDir "ep_idgl_outputs.pt"),
        "--abc-edges", $abcEdges,
        "--hic-bedpe", $testBedpe,
        "--hic-split-name", "test",
        "--topk", "$EvalTopK",
        "--edge-labels", $edgeLabels,
        "--edge-label-col", "edge_label_test",
        "--edge-metric-topks", $EdgeMetricTopKs,
        "--metrics-output", $metricsJson,
        "--ep-only"
    )
}

$metricsGlob = Join-Path $OutputRoot "seed*\eval_test_metrics.json"
$perSeedTsv = Join-Path $OutputRoot "multi_seed_metrics.tsv"
$summaryTsv = Join-Path $OutputRoot "multi_seed_metrics_summary.tsv"
Invoke-Step -Name "summarize" -LogPath (Join-Path $OutputRoot "summarize.log") -CommandArgs @(
    $summaryScript,
    "--metrics-glob", $metricsGlob,
    "--output-tsv", $perSeedTsv,
    "--summary-tsv", $summaryTsv
)

Write-Host ""
Write-Host "Multi-seed run complete."
Write-Host "Per-seed metrics: $perSeedTsv"
Write-Host "Mean/std summary: $summaryTsv"
