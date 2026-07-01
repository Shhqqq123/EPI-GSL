param(
    [string]$PythonExe = "python",
    [string]$ProjectRoot = "D:\Documents\Playground",
    [string]$PromoterPath = "D:\Documents\Playground\promoter_nodes_full.tsv",
    [string]$RePath = "D:\Documents\Playground\re_nodes_full.tsv",
    [string]$HicBedpe = "D:\Documents\Playground\ENCFF308MMM.bedpe\ENCFF308MMM.bedpe",
    [string]$Chrom = "chr5",
    [string]$Seeds = "1,2,3,4,5",
    [string]$ModelMode = "edge-rerank",
    [double]$TestFraction = 0.2,
    [int]$SampleSize = 5000,
    [int]$Epochs = 300,
    [double]$Lr = 0.0005,
    [double]$EdgeLossWeight = 1.0,
    [double]$NegativeRatio = 5.0,
    [string]$NegativeSampling = "abc-distance-matched",
    [double]$HardNegativeRatio = 0.0,
    [string]$RankingLossWeights = "0.1",
    [string]$HardRankLossWeights = "0.0",
    [string]$AbcRankLossWeights = "0.0",
    [string]$DeltaL2Weights = "0.001,0.01",
    [string]$DeltaLogitScales = "0.1,0.25,0.5",
    [double]$ValidationFraction = 0.0,
    [string]$ValidationMetric = "auprc",
    [int]$EarlyStoppingPatience = 0,
    [int]$EarlyStoppingMinEpochs = 0,
    [string]$LabelCol = "log1p_atac_signal_per_kb",
    [string]$EvalTopK = "1000",
    [string]$EdgeMetricTopKs = "500,1000,2000",
    [string]$WorkDir = "D:\Documents\Playground\work",
    [string]$GridRoot = "D:\Documents\Playground\outputs\edge_rerank_param_grid_chr5",
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Parse-DoubleList {
    param([string]$Text)
    if ([string]::IsNullOrWhiteSpace($Text)) { return @() }
    return @($Text -split "[,\s;]+" | Where-Object { $_ -ne "" } | ForEach-Object { [double]$_ })
}

function Format-ParamTag {
    param([double]$Value)
    $text = ("{0:g}" -f $Value)
    $text = $text -replace "-", "m"
    $text = $text -replace "\.", "p"
    return $text
}

function Invoke-Step {
    param(
        [string]$Name,
        [string]$LogPath,
        [string[]]$CommandArgs
    )
    Write-Host ""
    Write-Host "[$Name] $($CommandArgs -join ' ')"
    if ($DryRun) {
        return
    }
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $exe = $CommandArgs[0]
        $rest = @($CommandArgs | Select-Object -Skip 1)
        $output = & $exe @rest 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    $output | Tee-Object -FilePath $LogPath | Out-Host
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode"
    }
}

$seedScript = Join-Path $ProjectRoot "EPI-GSL\run_edge_seed_splits.ps1"
$gridSummaryScript = Join-Path $ProjectRoot "EPI-GSL\summarize_grid_metrics.py"
foreach ($path in @($seedScript, $gridSummaryScript, $PromoterPath, $RePath, $HicBedpe)) {
    if (-not (Test-Path $path)) {
        throw "Required path not found: $path"
    }
}

New-Item -ItemType Directory -Force -Path $GridRoot | Out-Null
$rankingList = Parse-DoubleList $RankingLossWeights
$hardRankingList = Parse-DoubleList $HardRankLossWeights
$abcRankingList = Parse-DoubleList $AbcRankLossWeights
$deltaL2List = Parse-DoubleList $DeltaL2Weights
$deltaScaleList = Parse-DoubleList $DeltaLogitScales

foreach ($rankingWeight in $rankingList) {
    foreach ($hardRankingWeight in $hardRankingList) {
        foreach ($abcRankingWeight in $abcRankingList) {
            foreach ($deltaL2 in $deltaL2List) {
                foreach ($deltaScale in $deltaScaleList) {
                $tag = "rank{0}_hard{1}_abc{2}_l2{3}_scale{4}" -f (Format-ParamTag $rankingWeight), (Format-ParamTag $hardRankingWeight), (Format-ParamTag $abcRankingWeight), (Format-ParamTag $deltaL2), (Format-ParamTag $deltaScale)
                $runRoot = Join-Path $GridRoot $tag
                New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

                $params = [ordered]@{
                    config = $tag
                    model_mode = $ModelMode
                    chrom = $Chrom
                    seeds = $Seeds
                    negative_sampling = $NegativeSampling
                    hard_negative_ratio = $HardNegativeRatio
                    ranking_loss_weight = $rankingWeight
                    hard_rank_loss_weight = $hardRankingWeight
                    abc_rank_loss_weight = $abcRankingWeight
                    delta_l2_weight = $deltaL2
                    delta_logit_scale = $deltaScale
                    validation_fraction = $ValidationFraction
                    validation_metric = $ValidationMetric
                    early_stopping_patience = $EarlyStoppingPatience
                    early_stopping_min_epochs = $EarlyStoppingMinEpochs
                    sample_size = $SampleSize
                    epochs = $Epochs
                    lr = $Lr
                }
                if (-not $DryRun) {
                    $params | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runRoot "grid_params.json") -Encoding UTF8
                }

                Invoke-Step -Name "grid $tag" -LogPath (Join-Path $runRoot "grid_run.log") -CommandArgs @(
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy", "Bypass",
                    "-File", $seedScript,
                    "-PythonExe", $PythonExe,
                    "-ProjectRoot", $ProjectRoot,
                    "-PromoterPath", $PromoterPath,
                    "-RePath", $RePath,
                    "-HicBedpe", $HicBedpe,
                    "-Chrom", $Chrom,
                    "-Seeds", $Seeds,
                    "-ModelMode", $ModelMode,
                    "-TestFraction", "$TestFraction",
                    "-SampleSize", "$SampleSize",
                    "-Epochs", "$Epochs",
                    "-Lr", "$Lr",
                    "-EdgeLossWeight", "$EdgeLossWeight",
                    "-NegativeRatio", "$NegativeRatio",
                    "-NegativeSampling", $NegativeSampling,
                    "-HardNegativeRatio", "$HardNegativeRatio",
                    "-RankingLossWeight", "$rankingWeight",
                    "-HardRankLossWeight", "$hardRankingWeight",
                    "-AbcRankLossWeight", "$abcRankingWeight",
                    "-DeltaL2Weight", "$deltaL2",
                    "-DeltaLogitScale", "$deltaScale",
                    "-ValidationFraction", "$ValidationFraction",
                    "-ValidationMetric", $ValidationMetric,
                    "-EarlyStoppingPatience", "$EarlyStoppingPatience",
                    "-EarlyStoppingMinEpochs", "$EarlyStoppingMinEpochs",
                    "-LabelCol", $LabelCol,
                    "-EvalTopK", $EvalTopK,
                    "-EdgeMetricTopKs", $EdgeMetricTopKs,
                    "-WorkDir", $WorkDir,
                    "-OutputRoot", $runRoot
                )
                }
            }
        }
    }
}

$gridSummary = Join-Path $GridRoot "grid_summary.tsv"
Invoke-Step -Name "summarize-grid" -LogPath (Join-Path $GridRoot "summarize_grid.log") -CommandArgs @(
    $PythonExe,
    $gridSummaryScript,
    "--grid-root", $GridRoot,
    "--output-tsv", $gridSummary
)

Write-Host ""
Write-Host "Grid run complete."
Write-Host "Grid summary: $gridSummary"
