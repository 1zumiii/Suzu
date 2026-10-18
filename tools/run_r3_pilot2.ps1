# 第三轮试点二：补生成一批语料（每域再 6500 行）、加大维基题量与降低领域句重复，再标注、短训、评测。
$ErrorActionPreference = "Continue"
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$py = Join-Path $repo ".venv\Scripts\python.exe"
$base = "D:\distill-data-r3"; $run = "$base\run2"; $p1 = "$base\pilot"; $p2 = "$base\pilot2"; $cor = "$base\pilot_merged"
[Console]::OutputEncoding = [Text.Encoding]::UTF8; $OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"; $env:HF_HOME = "D:\distill-data\hf"
$log = "$base\pilot2_run.log"
New-Item -ItemType Directory -Force $run, $cor | Out-Null
function Log($t) { Out-File -FilePath $log -Append -Encoding utf8 -InputObject $t }
function Step($name, $block) { Log "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss') >>> $name"; & $block 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Append -Encoding utf8; Log "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss') <<< $name exit=$LASTEXITCODE" }
foreach ($d in "chat", "tech", "anime") {
  Step "gen_$d" { & $py tools\gen_domain_corpus.py --out $p2 --domain $d --target 6500 --seed 2 }
  Step "filter_$d" {
    cmd /c copy /b "$p1\raw_$d.jsonl"+"$p2\raw_$d.jsonl" "$cor\raw_$d.jsonl"
    & $py tools\filter_domain_corpus.py --raw "$cor\raw_$d.jsonl" --out "$cor\$d.txt" }
}
Step "build" { & $py tools\r3_build.py --out $run --wiki-counts D:/distill-data-domain/word_counts.pkl --wiki-train D:/distill-data-r2/train_sentences.txt --wiki-items D:/distill-data-r2/items_train.jsonl --corpus-dir $cor --wiki-n 100000 --dom-items chat=20000 tech=12000 anime=8000 --mix "wiki=80,chat=10,tech=6,anime=4" }
New-Item -ItemType Directory -Force "$run\ckpt" | Out-Null
Copy-Item D:\distill-data-r2\ckpt\vocab.json "$run\ckpt\vocab.json"; Copy-Item D:\distill-data-r2\items_dev.jsonl, D:\distill-data-r2\meta.json, D:\distill-data-r2\gates.json $run
Step "label" { & $py tools\r3_label.py --data $run }
$env:DISTILL_DATA = $run; $env:DISTILL_ROUND = "2"; $env:DISTILL_TEACHER = "q35_4b_base"
$env:DISTILL_INIT = "D:\distill-data-r2\ckpt\student_best.pt"
$env:DISTILL_STUDENT_STEPS = "15000"; $env:DISTILL_STUDENT_LR = "1e-4"; $env:DISTILL_CKPT_EVERY = "1000"; $env:DISTILL_EVAL_EVERY = "3000"
Step "train" { & $py -c "import sys; sys.path.insert(0,'pipeline'); import s4_train; s4_train.run()" }
Remove-Item Env:DISTILL_ROUND, Env:DISTILL_DATA -ErrorAction SilentlyContinue
Step "eval" { & $py tools\r3_eval.py --ckpt "$run\ckpt\student_last.pt" --out "$run\eval_last.md" --name "Suzu-3 试点二（末步）" --holdout "$run\items_holdout.jsonl" }
Log "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss') ALL DONE"
