# Round 3 scale-up: scene seeds, then large corpus generation (anime, tech, chat). Resumable (raw files are appended to).
$ErrorActionPreference = "Continue"
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$py = Join-Path $repo ".venv\Scripts\python.exe"
$out = "D:\distill-data-r3\big"
New-Item -ItemType Directory -Force $out | Out-Null
[Console]::OutputEncoding = [Text.Encoding]::UTF8; $OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"; $env:HF_HOME = "D:\distill-data\hf"
$log = "D:\distill-data-r3\big_gen.log"
function Log($t) { Out-File -FilePath $log -Append -Encoding utf8 -InputObject $t }
function Step($name, $block) { Log "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss') >>> $name"; & $block 2>&1 | ForEach-Object { if ($_ -is [Management.Automation.ErrorRecord]) { $_.Exception.Message } else { "$_" } } | Out-File -FilePath $log -Append -Encoding utf8; Log "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss') <<< $name exit=$LASTEXITCODE" }
$targets = @{ anime = 62000; tech = 100000; chat = 190000 }
foreach ($d in "anime", "tech", "chat") {
  if (-not (Test-Path "$out\scenes_$d.jsonl")) { Step "scenes_$d" { & $py tools\gen_scenes.py --out "$out\scenes_$d.jsonl" --domain $d --rounds 2 } }
  Step "gen_$d" { & $py tools\gen_domain_corpus.py --out $out --domain $d --target $targets[$d] --seed 11 --batch 24 --scenes "$out\scenes_$d.jsonl" }
}
Log "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss') ALL DONE"
