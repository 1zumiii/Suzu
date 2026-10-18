# 后台守护：反复拉起 pipeline.py 直到全部完成、收到停止请求或连续失败过多。
param([string]$Smoke = "")
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$base = if ($env:DISTILL_DATA) { $env:DISTILL_DATA } else { Join-Path $root "data" }
New-Item -ItemType Directory -Force -Path (Join-Path $base "logs") | Out-Null
$log = Join-Path $base "logs\stdout.log"
$stop = Join-Path $base "STOP"
$env:PYTHONUTF8 = "1"
$fails = 0
while ($true) {
    if (Test-Path $stop) { break }
    $t0 = Get-Date
    cmd /c "`"$py`" -u pipeline\pipeline.py $Smoke >> `"$log`" 2>&1"
    $code = $LASTEXITCODE
    Add-Content -LiteralPath $log -Encoding UTF8 -Value ("[{0}] pipeline.py 退出码 {1}，本次运行 {2:N0} 秒" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $code, ((Get-Date) - $t0).TotalSeconds)
    if ($code -eq 0 -or $code -eq 75) { break }
    if (((Get-Date) - $t0).TotalMinutes -gt 10) { $fails = 0 }
    $fails++
    if ($fails -ge 30) { break }
    Start-Sleep -Seconds ([Math]::Min(300, 20 * $fails))
}
Remove-Item (Join-Path $base "watchdog.pid") -ErrorAction SilentlyContinue
