$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    npm.cmd ci --no-audit --no-fund
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    npm.cmd run build
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    .\.venv\Scripts\python.exe -m PyInstaller --noconfirm --onedir --console --hide-console hide-early --name LayerDeltaSpike --add-data 'static:static' --add-data 'worker.py:.' shell.py
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}
