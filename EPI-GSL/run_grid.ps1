param(
    [string]$PythonExe = "python",
    [string]$ProjectRoot = "D:\Documents\Playground",
    [string]$PromoterPath = "D:\Documents\Playground\promoter_nodes_full.tsv",
    [string]$RePath = "D:\Documents\Playground\re_nodes_full.tsv",
    [string]$HicBedpe = "D:\Documents\Playground\ENCFF308MMM.bedpe\ENCFF308MMM.bedpe",
    [string]$Chrom = "chr1",
    [int]$SampleSize = 5000,
    [int]$MaxDistance = 200000,
    [int]$Epochs = 500,
    [double]$Lr = 0.001,
    [double]$ReconWeight = 1.0,
    [double]$SparsityWeight = 1e-4,
    [double]$SmoothWeight = 1e-2,
    [string]$GraphAlphas = "0.7,0.8,0.9",
    [string]$TopKEdges = "30,50,80",
    [string]$StabilityWeights = "0.01,0.1",
    [string]$Seeds = "42,43,44",
    [string]$EvalTopK = "500,1000,2000,5000",
    [string]$GridOutputRoot = "D:\Documents\Playground\outputs\epi_gsl_grid",
    [switch]$DryRun
)

Set-StrictMode -Version Latest
# PyTorch/PyG warnings go to stderr but should not stop the script.
# We still stop on real failures via explicit exit-code checks and throws.
$ErrorActionPreference = "Stop"

$trainScript = Join-Path $ProjectRoot "EPI-GSL\train.py"
$evalScript = Join-Path $ProjectRoot "EPI-GSL\eval.py"

if (-not (Test-Path $trainScript)) {
    throw "train.py not found: $trainScript"
}
if (-not (Test-Path $evalScript)) {
    throw "eval.py not found: $evalScript"
}

New-Item -ItemType Directory -Force -Path $GridOutputRoot | Out-Null
$summaryPath = Join-Path $GridOutputRoot "grid_summary.tsv"
$summaryHeader = "run_id`tgraph_alpha`ttopk_edges`tstability_weight`tseed`teval_topk`tinit_hit_count`tinit_hit_rate`topt_hit_count`topt_hit_rate`tdelta_hit_rate`ttrain_output_dir"
Set-Content -Path $summaryPath -Value $summaryHeader -Encoding UTF8

function Parse-MetricLine {
    param([string]$Line)
    if ($Line -match "^topk=(\d+)\s+hit_count=(\d+)\s+hit_rate=([0-9eE\+\-\.]+)$") {
        return @{
            topk = [int]$Matches[1]
            hit_count = [int]$Matches[2]
            hit_rate = [double]$Matches[3]
        }
    }
    throw "Cannot parse metric line: $Line"
}

function Safe-Tag {
    param([double]$Value)
    return ($Value.ToString("0.########") -replace "\.", "p")
}

function Parse-DoubleList {
    param([string]$Text)
    if ([string]::IsNullOrWhiteSpace($Text)) { return @() }
    return @($Text -split "[,\s;]+" | Where-Object { $_ -ne "" } | ForEach-Object { [double]$_ })
}

function Parse-IntList {
    param([string]$Text)
    if ([string]::IsNullOrWhiteSpace($Text)) { return @() }
    return @($Text -split "[,\s;]+" | Where-Object { $_ -ne "" } | ForEach-Object { [int]$_ })
}

Set-Location $ProjectRoot
$GraphAlphaList = Parse-DoubleList $GraphAlphas
$TopKEdgesList = Parse-IntList $TopKEdges
$StabilityWeightList = Parse-DoubleList $StabilityWeights
$SeedList = Parse-IntList $Seeds
$EvalTopKList = Parse-IntList $EvalTopK

$totalRuns = $GraphAlphaList.Count * $TopKEdgesList.Count * $StabilityWeightList.Count * $SeedList.Count
$runIdx = 0

foreach ($ga in $GraphAlphaList) {
    foreach ($k in $TopKEdgesList) {
        foreach ($sw in $StabilityWeightList) {
            foreach ($seed in $SeedList) {
                $runIdx += 1
                $runId = "ga$(Safe-Tag $ga)_k${k}_sw$(Safe-Tag $sw)_s${seed}"
                $runDir = Join-Path $GridOutputRoot $runId
                New-Item -ItemType Directory -Force -Path $runDir | Out-Null

                Write-Host ""
                Write-Host "[$runIdx/$totalRuns] $runId"

                $trainArgs = @(
                    $trainScript,
                    "--promoter-path", $PromoterPath,
                    "--re-path", $RePath,
                    "--output-dir", $runDir,
                    "--sample-size", "$SampleSize",
                    "--max-distance", "$MaxDistance",
                    "--epochs", "$Epochs",
                    "--lr", "$Lr",
                    "--graph-alpha", "$ga",
                    "--topk-edges", "$k",
                    "--recon-weight", "$ReconWeight",
                    "--sparsity-weight", "$SparsityWeight",
                    "--smooth-weight", "$SmoothWeight",
                    "--stability-weight", "$sw",
                    "--seed", "$seed",
                    "--ep-only"
                )

                if ($Chrom -ne "") {
                    $trainArgs += @("--chrom", $Chrom)
                }

                if ($DryRun) {
                    Write-Host "[DryRun][Train] $PythonExe $($trainArgs -join ' ')"
                } else {
                    $trainLog = Join-Path $runDir "train.log"
                    $trainOutput = & $PythonExe @trainArgs 2>&1
                    $trainExitCode = $LASTEXITCODE
                    $trainOutput | Tee-Object -FilePath $trainLog | Out-Host
                    if ($trainExitCode -ne 0) {
                        throw "Training failed at run: $runId"
                    }
                }

                foreach ($evalK in $EvalTopKList) {
                    $evalTopKValue = [int]$evalK
                    $evalArgs = @(
                        $evalScript,
                        "--outputs-path", (Join-Path $runDir "ep_idgl_outputs.pt"),
                        "--hic-bedpe", $HicBedpe,
                        "--max-distance", "$MaxDistance",
                        "--topk", "$evalTopKValue",
                        "--ep-only"
                    )

                    if ($DryRun) {
                        Write-Host "[DryRun][Eval]  $PythonExe $($evalArgs -join ' ')"
                        continue
                    }

                    $evalLog = Join-Path $runDir ("eval_topk_{0}.log" -f $evalTopKValue)
                    $evalOutput = & $PythonExe @evalArgs 2>&1
                    $evalExitCode = $LASTEXITCODE
                    $evalOutput | Tee-Object -FilePath $evalLog | Out-Host
                    if ($evalExitCode -ne 0) {
                        throw "Evaluation failed at run: $runId, topk=$evalTopKValue"
                    }

                    $metricLines = @($evalOutput | Where-Object { $_ -match "^topk=\d+\s+hit_count=\d+\s+hit_rate=" })
                    if ($metricLines.Count -lt 2) {
                        throw "Cannot find two metric lines in eval output: run=$runId topk=$evalTopKValue"
                    }

                    $init = Parse-MetricLine -Line $metricLines[0]
                    $opt = Parse-MetricLine -Line $metricLines[1]
                    $delta = $opt.hit_rate - $init.hit_rate

                    $line = "{0}`t{1}`t{2}`t{3}`t{4}`t{5}`t{6}`t{7}`t{8}`t{9}`t{10}`t{11}" -f `
                        $runId, $ga, $k, $sw, $seed, $evalTopKValue, $init.hit_count, $init.hit_rate, $opt.hit_count, $opt.hit_rate, $delta, $runDir
                    Add-Content -Path $summaryPath -Value $line -Encoding UTF8
                }
            }
        }
    }
}

Write-Host ""
Write-Host "Grid search complete."
Write-Host "Summary: $summaryPath"
