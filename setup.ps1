# 一次性环境准备：建虚拟环境，安装 CUDA 12.8 版 torch 和其他依赖，最后跑预检。
# 用法：.\setup.ps1 [-Python "C:\Path\to\python.exe"]   需要 Python 3.12 或 3.13（torch cu128 暂无更新版本的轮子）
param([string]$Python = "")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    if (-not $Python) {
        foreach ($v in "3.12", "3.13") {
            if (Get-Command py -ErrorAction SilentlyContinue) {
                $p = & py "-$v" -c "import sys;print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $p) { $Python = $p; break }
            }
        }
    }
    if (-not $Python) { throw "找不到 Python 3.12/3.13。请安装后用 -Python 指定 python.exe 的完整路径。" }
    $ver = & $Python -c "import sys;print('%d.%d' % sys.version_info[:2])"
    if ($ver -notin @("3.12", "3.13")) { throw "Python $ver 不受 torch cu128 支持，请改用 3.12 或 3.13（$Python）" }
    & $Python -m venv .venv
}
$py = Join-Path $root ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r requirements.txt
& $py preflight.py
Write-Host "`n环境就绪。先跑冒烟测试：.\run.ps1 smoke"
