param(
    [string]$PythonExe = "python",
    [string]$ProjectRoot = "D:\Documents\Playground",
    [string]$PromoterPath = "D:\Documents\Playground\promoter_nodes_full.tsv",
    [string]$RePath = "D:\Documents\Playground\re_nodes_full.tsv",
    [string]$HicBedpe = "D:\Documents\Playground\ENCFF308MMM.bedpe\ENCFF308MMM.bedpe",
    [string]$TrainChroms = "chr1,chr2,chr3,chr4,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22",
    [string]$TestChrom = "chr5",
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
    [string]$NegativeSampling = "abc-distance-matched",
    [double]$RankingLossWeight = 0.1,
    [double]$DeltaL2Weight = 0.001,
    [double]$DeltaLogitScale = 0.25,
    [string]$LabelCol = "log1p_atac_signal_per_kb",
    [int]$EvalTopK = 1000,
    [string]$EdgeMetricTopKs = "500,1000,2000",
    [string]$WorkDir = "D:\Documents\Playground\work",
    [string]$OutputRoot = "D:\Documents\Playground\outputs\loco_chr5_train_chr1_4_chr6_22",
    [switch]$RegenerateAbcEdges,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Parse-ChromList {
    param([string]$Text)
    if ([string]::IsNullOrWhiteSpace($Text)) { return @() }
    return @($Text -split "[,\s;]+" | Where-Object { $_ -ne "" })
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
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & $PythonExe @CommandArgs 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    $output | Tee-Object -FilePath $LogPath | Out-Host
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode"
    }
}

$makeAbcScript = Join-Path $ProjectRoot "EPI-GSL\make_abc_edges.py"
$concatScript = Join-Path $ProjectRoot "EPI-GSL\concat_tsv.py"
$makeLabelsScript = Join-Path $ProjectRoot "EPI-GSL\make_edge_labels.py"
$trainScript = Join-Path $ProjectRoot "EPI-GSL\train.py"
$predictScript = Join-Path $ProjectRoot "EPI-GSL\predict_heldout_chrom.py"
$evalScript = Join-Path $ProjectRoot "EPI-GSL\eval.py"

foreach ($path in @($makeAbcScript, $concatScript, $makeLabelsScript, $trainScript, $predictScript, $evalScript, $PromoterPath, $RePath, $HicBedpe)) {
    if (-not (Test-Path $path)) {
        throw "Required path not found: $path"
    }
}

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
Set-Location $ProjectRoot

$trainChromList = Parse-ChromList $TrainChroms
$allChromList = @($trainChromList + @($TestChrom) | Select-Object -Unique)

foreach ($chrom in $allChromList) {
    $abcPath = Join-Path $WorkDir ("abc_edges_{0}.tsv" -f $chrom)
    if ($RegenerateAbcEdges -or -not (Test-Path $abcPath)) {
        Invoke-Step -Name "make-abc $chrom" -LogPath (Join-Path $OutputRoot ("make_abc_{0}.log" -f $chrom)) -CommandArgs @(
            $makeAbcScript,
            "--promoter-path", $PromoterPath,
            "--re-path", $RePath,
            "--output-path", $abcPath,
            "--chrom", $chrom
        )
    } else {
        Write-Host "Using existing ABC edges: $abcPath"
    }
}

$trainAbcEdges = Join-Path $WorkDir ("abc_edges_train_excluding_{0}.tsv" -f $TestChrom)
$trainAbcInputs = @()
foreach ($chrom in $trainChromList) {
    $trainAbcInputs += (Join-Path $WorkDir ("abc_edges_{0}.tsv" -f $chrom))
}

Invoke-Step -Name "concat-train-abc" -LogPath (Join-Path $OutputRoot "concat_train_abc.log") -CommandArgs @(
    @($concatScript, "--inputs") + $trainAbcInputs + @("--output", $trainAbcEdges)
)

$trainEdgeLabels = Join-Path $WorkDir ("edge_labels_train_excluding_{0}.tsv" -f $TestChrom)
Invoke-Step -Name "label-train-edges" -LogPath (Join-Path $OutputRoot "label_train_edges.log") -CommandArgs @(
    $makeLabelsScript,
    "--abc-edges", $trainAbcEdges,
    "--hic-bedpe", $HicBedpe,
    "--output-path", $trainEdgeLabels
)

$testAbcEdges = Join-Path $WorkDir ("abc_edges_{0}.tsv" -f $TestChrom)
$testEdgeLabels = Join-Path $WorkDir ("edge_labels_{0}_all.tsv" -f $TestChrom)
Invoke-Step -Name "label-test-edges" -LogPath (Join-Path $OutputRoot "label_test_edges.log") -CommandArgs @(
    $makeLabelsScript,
    "--abc-edges", $testAbcEdges,
    "--hic-bedpe", $HicBedpe,
    "--output-path", $testEdgeLabels,
    "--chrom", $TestChrom
)

$trainDir = Join-Path $OutputRoot "train"
$predDir = Join-Path $OutputRoot ("pred_{0}" -f $TestChrom)
New-Item -ItemType Directory -Force -Path $trainDir | Out-Null
New-Item -ItemType Directory -Force -Path $predDir | Out-Null

Invoke-Step -Name "train-loco" -LogPath (Join-Path $OutputRoot "train.log") -CommandArgs @(
    $trainScript,
    "--model-mode", "edge-rerank",
    "--promoter-path", $PromoterPath,
    "--re-path", $RePath,
    "--output-dir", $trainDir,
    "--abc-edges", $trainAbcEdges,
    "--edge-labels", $trainEdgeLabels,
    "--edge-label-col", "edge_label",
    "--include-chroms", $TrainChroms,
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
    "--negative-sampling", $NegativeSampling,
    "--ranking-loss-weight", "$RankingLossWeight",
    "--delta-l2-weight", "$DeltaL2Weight",
    "--delta-logit-scale", "$DeltaLogitScale",
    "--ep-only"
)

Invoke-Step -Name "predict-heldout-$TestChrom" -LogPath (Join-Path $OutputRoot "predict_heldout.log") -CommandArgs @(
    $predictScript,
    "--model-path", (Join-Path $trainDir "ep_idgl_model.pt"),
    "--train-bundle", (Join-Path $trainDir "ep_idgl_outputs.pt"),
    "--run-config", (Join-Path $trainDir "run_config.json"),
    "--promoter-path", $PromoterPath,
    "--re-path", $RePath,
    "--output-dir", $predDir,
    "--abc-edges", $testAbcEdges,
    "--edge-table", $testEdgeLabels,
    "--chrom", $TestChrom,
    "--sample-size", "$SampleSize"
)

Invoke-Step -Name "eval-heldout-$TestChrom" -LogPath (Join-Path $OutputRoot "eval_heldout.log") -CommandArgs @(
    $evalScript,
    "--outputs-path", (Join-Path $predDir "ep_idgl_outputs.pt"),
    "--abc-edges", $testAbcEdges,
    "--hic-bedpe", $HicBedpe,
    "--hic-split-name", $TestChrom,
    "--topk", "$EvalTopK",
    "--edge-labels", $testEdgeLabels,
    "--edge-label-col", "edge_label",
    "--edge-metric-topks", $EdgeMetricTopKs,
    "--metrics-output", (Join-Path $predDir "eval_metrics.json"),
    "--ep-only"
)

Write-Host ""
Write-Host "LOCO run complete."
Write-Host "Train output: $trainDir"
Write-Host "Held-out prediction output: $predDir"
Write-Host "Metrics JSON: $(Join-Path $predDir 'eval_metrics.json')"
