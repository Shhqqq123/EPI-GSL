param(
    [string]$PythonExe = "python",
    [string]$ProjectRoot = "D:\Documents\Playground",
    [string]$PromoterPath = "D:\Documents\Playground\promoter_nodes_full.tsv",
    [string]$RePath = "D:\Documents\Playground\re_nodes_full.tsv",
    [string]$HicBedpe = "D:\Documents\Playground\ENCFF308MMM.bedpe\ENCFF308MMM.bedpe",
    [string]$CandidateChroms = "chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22",
    [string]$TestChroms = "chr1,chr5,chr10,chr15,chr20",
    [string]$ModelMode = "sparse-gsl",
    [switch]$ChromBatchTraining,
    [int]$SampleSize = 10000,
    [int]$Epochs = 300,
    [double]$Lr = 0.0005,
    [int]$GraphIters = 2,
    [double]$EdgeLossWeight = 1.0,
    [double]$NegativeRatio = 5.0,
    [string]$NegativeSampling = "abc-distance-matched",
    [double]$HardNegativeRatio = 2.0,
    [double]$RankingLossWeight = 0.1,
    [double]$HardRankLossWeight = 0.1,
    [double]$HardRankMargin = 0.5,
    [int]$HardRankNegativesPerPositive = 4,
    [int]$HardRankMaxPairs = 50000,
    [double]$HardRankTopNegativeRatio = 20.0,
    [double]$AbcRankLossWeight = 0.0,
    [double]$AbcRankMargin = 0.0,
    [int]$AbcRankMaxPairs = 50000,
    [double]$AbcRankMinScoreGap = 0.0,
    [string]$AbcRankScope = "negatives",
    [double]$DeltaL2Weight = 0.003,
    [double]$DeltaLogitScale = 0.075,
    [double]$ValidationFraction = 0.0,
    [string]$ValidationChroms = "",
    [switch]$AutoValidationChrom,
    [int]$AutoValidationChromCount = 1,
    [string]$AutoValidationChromStrategy = "next",
    [string]$ValidationMetric = "auprc",
    [int]$EarlyStoppingPatience = 0,
    [int]$EarlyStoppingMinEpochs = 0,
    [double]$ScoreBlendAlpha = -1.0,
    [string]$ScoreBlendAlphas = "",
    [string]$ScoreBlendMethod = "rank",
    [string]$ScoreBlendMetric = "auprc",
    [string]$LabelCol = "log1p_atac_signal_per_kb",
    [int]$EvalTopK = 1000,
    [string]$EdgeMetricTopKs = "500,1000,2000",
    [string]$WorkDir = "D:\Documents\Playground\work",
    [string]$OutputRoot = "D:\Documents\Playground\outputs\loco_multi_sparse_gsl_hardrank",
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

function Select-AutoValidationChroms {
    param(
        [string[]]$CandidateList,
        [string[]]$TrainChroms,
        [string]$TestChrom,
        [int]$Count,
        [string]$Strategy
    )
    if ($Count -le 0 -or $TrainChroms.Count -eq 0) {
        return @()
    }

    $trainSet = @{}
    foreach ($chrom in $TrainChroms) {
        $trainSet[$chrom] = $true
    }

    $ordered = @()
    $candidateCount = $CandidateList.Count
    $testIndex = [Array]::IndexOf($CandidateList, $TestChrom)
    if ($testIndex -lt 0) {
        $testIndex = 0
    }

    if ($Strategy -eq "first") {
        $ordered = @($TrainChroms)
    } elseif ($Strategy -eq "previous") {
        for ($offset = 1; $offset -le $candidateCount; $offset++) {
            $idx = ($testIndex - $offset) % $candidateCount
            if ($idx -lt 0) {
                $idx += $candidateCount
            }
            $chrom = $CandidateList[$idx]
            if ($trainSet.ContainsKey($chrom)) {
                $ordered += $chrom
            }
        }
    } else {
        if ($Strategy -ne "next") {
            throw "Unknown AutoValidationChromStrategy: $Strategy. Use next, previous, or first."
        }
        for ($offset = 1; $offset -le $candidateCount; $offset++) {
            $idx = ($testIndex + $offset) % $candidateCount
            $chrom = $CandidateList[$idx]
            if ($trainSet.ContainsKey($chrom)) {
                $ordered += $chrom
            }
        }
    }

    return @($ordered | Select-Object -First $Count)
}

function Invoke-Step {
    param(
        [string]$Name,
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
    $logPath = Join-Path $OutputRoot ("{0}.log" -f ($Name -replace "[^\w.-]+", "_"))
    $output | Tee-Object -FilePath $logPath | Out-Host
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode"
    }
}

$locoScript = Join-Path $ProjectRoot "EPI-GSL\run_loco_chrom_split.ps1"
$summaryScript = Join-Path $ProjectRoot "EPI-GSL\summarize_loco_metrics.py"
foreach ($path in @($locoScript, $summaryScript, $PromoterPath, $RePath, $HicBedpe)) {
    if (-not (Test-Path $path)) {
        throw "Required path not found: $path"
    }
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$candidateList = Parse-ChromList $CandidateChroms
$testList = Parse-ChromList $TestChroms

foreach ($testChrom in $testList) {
    $trainChroms = @($candidateList | Where-Object { $_ -ne $testChrom })
    if ($trainChroms.Count -eq 0) {
        throw "No training chromosomes left after excluding $testChrom"
    }
    $runRoot = Join-Path $OutputRoot ("heldout_{0}" -f $testChrom)
    $trainChromText = $trainChroms -join ","
    $validationChromText = $ValidationChroms
    if ($AutoValidationChrom -and [string]::IsNullOrWhiteSpace($validationChromText)) {
        $autoValidationChroms = Select-AutoValidationChroms `
            -CandidateList $candidateList `
            -TrainChroms $trainChroms `
            -TestChrom $testChrom `
            -Count $AutoValidationChromCount `
            -Strategy $AutoValidationChromStrategy
        $validationChromText = $autoValidationChroms -join ","
    }

    $args = @(
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $locoScript,
        "-PythonExe", $PythonExe,
        "-ProjectRoot", $ProjectRoot,
        "-PromoterPath", $PromoterPath,
        "-RePath", $RePath,
        "-HicBedpe", $HicBedpe,
        "-TrainChroms", $trainChromText,
        "-TestChrom", $testChrom,
        "-ModelMode", $ModelMode,
        "-SampleSize", "$SampleSize",
        "-Epochs", "$Epochs",
        "-Lr", "$Lr",
        "-GraphIters", "$GraphIters",
        "-EdgeLossWeight", "$EdgeLossWeight",
        "-NegativeRatio", "$NegativeRatio",
        "-NegativeSampling", $NegativeSampling,
        "-HardNegativeRatio", "$HardNegativeRatio",
        "-RankingLossWeight", "$RankingLossWeight",
        "-HardRankLossWeight", "$HardRankLossWeight",
        "-HardRankMargin", "$HardRankMargin",
        "-HardRankNegativesPerPositive", "$HardRankNegativesPerPositive",
        "-HardRankMaxPairs", "$HardRankMaxPairs",
        "-HardRankTopNegativeRatio", "$HardRankTopNegativeRatio",
        "-AbcRankLossWeight", "$AbcRankLossWeight",
        "-AbcRankMargin", "$AbcRankMargin",
        "-AbcRankMaxPairs", "$AbcRankMaxPairs",
        "-AbcRankMinScoreGap", "$AbcRankMinScoreGap",
        "-AbcRankScope", $AbcRankScope,
        "-DeltaL2Weight", "$DeltaL2Weight",
        "-DeltaLogitScale", "$DeltaLogitScale",
        "-ValidationFraction", "$ValidationFraction",
        "-ValidationChroms", $validationChromText,
        "-ValidationMetric", $ValidationMetric,
        "-EarlyStoppingPatience", "$EarlyStoppingPatience",
        "-EarlyStoppingMinEpochs", "$EarlyStoppingMinEpochs",
        "-ScoreBlendAlpha", "$ScoreBlendAlpha",
        "-ScoreBlendAlphas", $ScoreBlendAlphas,
        "-ScoreBlendMethod", $ScoreBlendMethod,
        "-ScoreBlendMetric", $ScoreBlendMetric,
        "-LabelCol", $LabelCol,
        "-EvalTopK", "$EvalTopK",
        "-EdgeMetricTopKs", $EdgeMetricTopKs,
        "-WorkDir", $WorkDir,
        "-OutputRoot", $runRoot
    )
    if ($RegenerateAbcEdges) {
        $args += "-RegenerateAbcEdges"
    }
    if ($ChromBatchTraining) {
        $args += "-ChromBatchTraining"
    }
    if ($DryRun) {
        $args += "-DryRun"
    }
    Invoke-Step -Name "loco-$testChrom" -CommandArgs $args
}

$summaryPath = Join-Path $OutputRoot "loco_summary.tsv"
Invoke-Step -Name "summarize-loco" -CommandArgs @(
    $PythonExe,
    $summaryScript,
    "--output-root", $OutputRoot,
    "--output-tsv", $summaryPath
)

Write-Host ""
Write-Host "Multi-LOCO run complete."
Write-Host "Summary: $summaryPath"
