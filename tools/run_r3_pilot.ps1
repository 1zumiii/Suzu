# 第三轮试点：4B-Base 标注 → 从 Suzu-2 起步的短训 → 评测。可断点续跑（标注与训练都会续）。
$ErrorActionPreference = "Continue"
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$run = "D:\distill-data-r3\run"
$py = Join-Path $repo ".venv\Scripts\python.exe"
[Console]::OutputEncoding = [Text.Encoding]::UTF8; $OutputEncoding = [Text.Encoding]::UTF8; $env:PYTHONIOENCODING = "utf-8"; $env:HF_HOME = "D:\distill-data\hf"
$log = "D:\distill-data-r3\pilot_run.log"
function Log($t) { Out-File -FilePath $log -Append -Encoding utf8 -InputObject $t }   # PowerShell 5.1 的 *>> 会写成 UTF-16，统一用 Out-File 写 UTF-8
function Step($name, $block) { Log "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss') >>> $name"; & $block 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Append -Encoding utf8; Log "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss') <<< $name exit=$LASTEXITCODE" }
Step "label" { & $py tools\r3_label.py --data $run }
$env:DISTILL_DATA = $run; $env:DISTILL_ROUND = "2"; $env:DISTILL_TEACHER = "q35_4b_base"
$env:DISTILL_INIT = "D:\distill-data-r2\ckpt\student_best.pt"
$env:DISTILL_STUDENT_STEPS = "12000"; $env:DISTILL_STUDENT_LR = "2e-4"; $env:DISTILL_CKPT_EVERY = "1000"; $env:DISTILL_EVAL_EVERY = "3000"
Step "train" { & $py -c "import sys; sys.path.insert(0,'pipeline'); import s4_train; s4_train.run()" }
Remove-Item Env:DISTILL_ROUND, Env:DISTILL_DATA -ErrorAction SilentlyContinue
Step "eval_last" { & $py tools\r3_eval.py --ckpt "$run\ckpt\student_last.pt" --out "$run\eval_last.md" --name "Suzu-3试点（末步）" }
Step "eval_best" { & $py tools\r3_eval.py --ckpt "$run\ckpt\student_best.pt" --out "$run\eval_best.md" --name "Suzu-3试点（开发集最优）" }
Log "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss') ALL DONE"
