param([switch]$RendererOnly)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    npm.cmd ci --no-audit --no-fund
    if ($LASTEXITCODE) {exit $LASTEXITCODE}
    npm.cmd run build
    if ($LASTEXITCODE) {exit $LASTEXITCODE}
    if (!$RendererOnly) {
        py -3.12 -m venv .build-env
        if ($LASTEXITCODE) {exit $LASTEXITCODE}
        .\.build-env\Scripts\python.exe -m pip install -r ..\spikes\desktop\requirements.txt
        if ($LASTEXITCODE) {exit $LASTEXITCODE}
        .\.build-env\Scripts\python.exe -m PyInstaller --noconfirm --onedir --console --hide-console hide-early --icon public/layerdelta.ico --distpath build\package --workpath build\pyinstaller --add-data 'dist:dist' --name LayerDelta launcher.py
        if ($LASTEXITCODE) {exit $LASTEXITCODE}
    }
} finally {
    Pop-Location
}
