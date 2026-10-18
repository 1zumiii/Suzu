# Round 3 continuation training (Suzu-3): build items from the scaled corpus, label with 4B-Base, continue from Suzu-2, evaluate. Resumable.
$ErrorActionPreference = "Continue"
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$py = Join-Path $repo ".venv\Scripts\python.exe"
$big = "D:\distill-data-r3\big"; $run = "D:\distill-data-r3\run3"
[Console]::OutputEncoding = [Text.Encoding]::UTF8; $OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"; $env:HF_HOME = "D:\distill-data\hf"
$log = "D:\distill-data-r3\run3.log"
function Log($t) { Out-File -FilePath $log -Append -Encoding utf8 -InputObject $t }
function Step($name, $block) { Log "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss') >>> $name"; & $block 2>&1 | ForEach-Object { if ($_ -is [Management.Automation.ErrorRecord]) { $_.Exception.Message } else { "$_" } } | Out-File -FilePath $log -Append -Encoding utf8; Log "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss') <<< $name exit=$LASTEXITCODE" }
New-Item -ItemType Directory -Force "$run\ckpt" | Out-Null
if (-not (Test-Path "$run\items_train.jsonl")) {
  Step "build" { & $py tools\r3_build.py --out $run --wiki-counts D:/distill-data-domain/word_counts.pkl --wiki-train D:/distill-data-r2/train_sentences.txt --wiki-items D:/distill-data-r2/items_train.jsonl --corpus-dir $big --wiki-n 100000 --dom-items chat=20000 tech=12000 anime=8000 --wiki-sent-n 936000 --mix "wiki=80,chat=10,tech=6,anime=4" }
  Copy-Item D:\distill-data-r2\ckpt\vocab.json "$run\ckpt\vocab.json" -Force
  Copy-Item D:\distill-data-r2\items_dev.jsonl, D:\distill-data-r2\meta.json, D:\distill-data-r2\gates.json $run -Force
}
Step "label" { & $py tools\r3_label.py --data $run }
$env:DISTILL_DATA = $run; $env:DISTILL_ROUND = "2"; $env:DISTILL_TEACHER = "q35_4b_base"
$env:DISTILL_INIT = "D:\distill-data-r2\ckpt\student_best.pt"
$env:DISTILL_STUDENT_STEPS = "15000"; $env:DISTILL_STUDENT_LR = "1e-4"; $env:DISTILL_CKPT_EVERY = "1000"; $env:DISTILL_EVAL_EVERY = "3000"
Step "train" { & $py -c "import sys; sys.path.insert(0,'pipeline'); import s4_train; s4_train.run()" }
Remove-Item Env:DISTILL_ROUND, Env:DISTILL_DATA -ErrorAction SilentlyContinue
Step "eval_suzu3" { & $py tools\r3_eval.py --ckpt "$run\ckpt\student_last.pt" --out "$run\eval_suzu3.md" --name "Suzu-3 (continued, final step)" --holdout "$run\items_holdout.jsonl" }
Step "eval_pilot2" { & $py tools\r3_eval.py --ckpt "D:\distill-data-r3\run2\ckpt\student_last.pt" --out "$run\eval_pilot2.md" --name "Pilot 2 (final step)" --holdout "$run\items_holdout.jsonl" }
Log "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss') ALL DONE"
