param([string]$OutDir = (Join-Path $PSScriptRoot '..\dist'))
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '..\desktop\build.ps1')
if ($LASTEXITCODE) {exit $LASTEXITCODE}
& (Join-Path $PSScriptRoot '..\desktop\.build-env\Scripts\python.exe') (Join-Path $PSScriptRoot 'build.py') --out $OutDir
exit $LASTEXITCODE
