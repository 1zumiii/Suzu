# 第二轮训练启动脚本：设置第二轮所需的环境变量，再调用 run.ps1 start（后台运行，可 run.ps1 status / stop 管理）。
# 用法：.\run_round2.ps1 -Data <第二轮数据目录> -Round1 <第一轮数据目录>
# 第二轮沿用第一轮的评测题、开发题、基线打分与 HF 缓存，只新增语料、训练题、4B 标注、学生与四字评测。
param(
    [Parameter(Mandatory = $true)][string]$Data,
    [Parameter(Mandatory = $true)][string]$Round1
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force -Path (Join-Path $Data "scores"), (Join-Path $Data "state") | Out-Null
foreach ($f in "items_eval.jsonl", "items_dev.jsonl", "meta.json", "gates.json") {
    if (-not (Test-Path (Join-Path $Data $f))) { Copy-Item (Join-Path $Round1 $f) $Data }
}
foreach ($f in "ngram", "q35_0p8b", "q35_4b", "q35_9b") {
    $dst = Join-Path $Data "scores\$f.jsonl"
    if (-not (Test-Path $dst)) { Copy-Item (Join-Path $Round1 "scores\$f.jsonl") $dst }
}
$r1s = Join-Path $Data "scores\student_r1.jsonl"
if (-not (Test-Path $r1s)) { Copy-Item (Join-Path $Round1 "scores\student.jsonl") $r1s }
$env:DISTILL_ROUND = "2"
$env:DISTILL_TEACHER = "q35_4b"
$env:DISTILL_DATA = $Data
$env:HF_HOME = Join-Path $Round1 "hf"
& (Join-Path $root "run.ps1") start
