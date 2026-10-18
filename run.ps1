# 启停与查看。用法：.\run.ps1 <start|stop|status|smoke|retry|autostart|autostart-off|tail>
param([Parameter(Position = 0)][string]$Cmd = "status")
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$base = if ($env:DISTILL_DATA) { $env:DISTILL_DATA } else { Join-Path $root "data" }
New-Item -ItemType Directory -Force -Path $base | Out-Null
$pidFile = Join-Path $base "watchdog.pid"
$stopFile = Join-Path $base "STOP"

function Get-Running {
    if (Test-Path $pidFile) {
        $p = Get-Process -Id ([int](Get-Content $pidFile)) -ErrorAction SilentlyContinue
        if ($p) { return $p }
    }
    return $null
}

switch ($Cmd) {
    "start" {
        if (Get-Running) { Write-Host "已在运行（PID $((Get-Running).Id)）"; break }
        Remove-Item $stopFile -ErrorAction SilentlyContinue
        $p = Start-Process powershell -WindowStyle Hidden -PassThru -ArgumentList @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $root "watchdog.ps1"))
        Set-Content $pidFile $p.Id
        Write-Host "已在后台启动（PID $($p.Id)）。关掉窗口、注销都不影响；查看进度：.\run.ps1 status"
    }
    "stop" {
        New-Item -ItemType File -Force -Path $stopFile | Out-Null
        $p = Get-Running
        if (-not $p) { Write-Host "没有在运行"; break }
        Write-Host "已发出停止请求，等待当前批次存盘…"
        for ($i = 0; $i -lt 90 -and -not $p.HasExited; $i++) { Start-Sleep 2 }
        if (-not $p.HasExited) {
            Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*pipeline.py*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
            Stop-Process -Id $p.Id -Force
            Write-Host "超时，已强制结束（最多损失约 1 分钟进度，下次启动自动续上）"
        } else { Write-Host "已停止，进度已保存" }
        Remove-Item $pidFile -ErrorAction SilentlyContinue
    }
    "status" {
        $p = Get-Running
        Write-Host ("状态：" + $(if ($p) { "运行中（PID $($p.Id)）" } else { "未运行" }))
        $state = Join-Path $base "state"
        if (Test-Path $state) {
            Get-ChildItem $state -Filter "*.done" | ForEach-Object { Write-Host ("  完成 " + $_.BaseName) }
            Get-ChildItem $state -Filter "*.failed" | ForEach-Object { Write-Host ("  失败 " + $_.BaseName) }
            Get-ChildItem $state -Filter "*.skipped" | ForEach-Object { Write-Host ("  跳过 " + $_.BaseName) }
            $pr = Join-Path $state "progress.json"
            if (Test-Path $pr) { Write-Host ("当前进度：" + (Get-Content $pr -Raw)) }
        }
        if (Test-Path (Join-Path $base "DONE.txt")) { Write-Host "全部完成：见 data\report.md（或 report_stage0.md）" }
        $lg = Join-Path $base "logs\pipeline.log"
        if (Test-Path $lg) { Write-Host "--- 最近日志 ---"; Get-Content $lg -Tail 12 }
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
        }
    }
    "tail" { Get-Content (Join-Path $base "logs\pipeline.log") -Tail 40 -Wait }
    "smoke" { & $py pipeline\pipeline.py --smoke }
    "retry" {
        Get-ChildItem (Join-Path $base "state") -Filter "*.failed" | Remove-Item
        Remove-Item (Join-Path $base "state\failures.json") -ErrorAction SilentlyContinue
        Write-Host "已清除失败标记，执行 .\run.ps1 start 重试"
    }
    "autostart" {
        schtasks /Create /TN "RimeDistill" /SC ONLOGON /RL LIMITED /F /TR "powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$root\run.ps1`" start"
        Write-Host "已登记登录后自动续跑（重启、断电、系统更新后无需手动操作）"
    }
    "autostart-off" { schtasks /Delete /TN "RimeDistill" /F }
    default { Write-Host "未知命令 $Cmd" }
}
