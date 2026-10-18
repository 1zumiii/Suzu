# Round 3 scale-up, step 2: first filter (dedupe, charset, leak check), perplexity scoring with 4B-Base, final filter. Resumable.
$ErrorActionPreference = "Continue"
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$py = Join-Path $repo ".venv\Scripts\python.exe"
$big = "D:\distill-data-r3\big"
[Console]::OutputEncoding = [Text.Encoding]::UTF8; $OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"; $env:HF_HOME = "D:\distill-data\hf"
$log = "D:\distill-data-r3\big_filter.log"
function Log($t) { Out-File -FilePath $log -Append -Encoding utf8 -InputObject $t }
function Step($name, $block) { Log "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss') >>> $name"; & $block 2>&1 | ForEach-Object { if ($_ -is [Management.Automation.ErrorRecord]) { $_.Exception.Message } else { "$_" } } | Out-File -FilePath $log -Append -Encoding utf8; Log "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss') <<< $name exit=$LASTEXITCODE" }
$hold = @(Get-ChildItem "$repo\tests\holdout\*.txt" | ForEach-Object { $_.FullName })
$drop = @{ anime = 0.10; tech = 0.06; chat = 0.05 }
foreach ($d in "anime", "tech", "chat") {
  if (-not (Test-Path "$big\${d}_f1.txt")) { Step "filter1_$d" { & $py tools\filter_domain_corpus.py --raw "$big\raw_$d.jsonl" --out "$big\${d}_f1.txt" --exclude $hold } }
}
foreach ($d in "anime", "tech", "chat") {
  Step "ppl_$d" { & $py tools\r3_ppl.py --sentences "$big\${d}_f1.txt" --out "$big\ppl_$d.tsv" }
}
foreach ($d in "anime", "tech", "chat") {
  Step "filter2_$d" { & $py tools\filter_domain_corpus.py --raw "$big\raw_$d.jsonl" --out "$big\$d.txt" --exclude $hold --ppl "$big\ppl_$d.tsv" --ppl-drop $drop[$d] }
}
Log "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss') ALL DONE"
