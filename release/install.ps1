param(
    [Parameter(Mandatory=$true)][ValidateSet('cpu','cuda')][string]$ProfileName,
    [switch]$DesktopShortcut
)
$ErrorActionPreference = 'Stop'
if (![Environment]::Is64BitOperatingSystem) {throw 'LayerDelta requires Windows x64.'}
Push-Location $PSScriptRoot
try {
    $manifest = Get-Content -Raw -LiteralPath 'package.json' | ConvertFrom-Json
    foreach ($file in $manifest.files.PSObject.Properties) {
        if ((Get-FileHash -LiteralPath $file.Name -Algorithm SHA256).Hash -ne $file.Value) {
            throw "Package file failed verification: $($file.Name)"
        }
    }
    $edgeId = '{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'
    $keys = "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\$edgeId","HKCU:\Software\Microsoft\EdgeUpdate\Clients\$edgeId"
    $edge = Get-ItemProperty -LiteralPath $keys -ErrorAction SilentlyContinue | Where-Object { $_.pv -and ([version]$_.pv -gt [version]'0.0.0.0') }
    if (!$edge) {
        $bootstrap = Join-Path ([IO.Path]::GetTempPath()) ("layerdelta-webview2-" + [guid]::NewGuid() + ".exe")
        try {
            Invoke-WebRequest -UseBasicParsing -Uri 'https://go.microsoft.com/fwlink/p/?LinkId=2124703' -OutFile $bootstrap
            $sig = Get-AuthenticodeSignature -LiteralPath $bootstrap
            if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notmatch 'CN=Microsoft Corporation(,|$)') {
                throw 'WebView2 installer signature verification failed.'
            }
            $installer = Start-Process -FilePath $bootstrap -ArgumentList '/silent','/install' -Wait -PassThru -WindowStyle Hidden
            if ($installer.ExitCode) {throw "WebView2 installer exited with $($installer.ExitCode)"}
        } finally {
            if (Test-Path -LiteralPath $bootstrap) {Remove-Item -LiteralPath $bootstrap}
        }
    }
    $marker = 'runtime/profile.txt'
    if ((Test-Path -LiteralPath $marker) -and (Get-Content -Raw $marker).Trim() -ne $ProfileName) {
        throw 'Extract the ZIP into a separate folder for the other runtime profile.'
    }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot 'python'
    $env:UV_LINK_MODE = 'copy'
    $uv = Join-Path $PSScriptRoot 'tools/uv.exe'
    & $uv --no-config python install 3.12.5 --no-bin
    if ($LASTEXITCODE) {throw 'Managed Python download failed.'}
    $python = Join-Path $PSScriptRoot 'runtime/Scripts/python.exe'
    if (!(Test-Path -LiteralPath $python)) {
        & $uv --no-config venv --managed-python --python 3.12.5 runtime
        if ($LASTEXITCODE) {throw 'Runtime environment creation failed.'}
    }
    & $uv --no-config pip install --python $python --require-hashes --only-binary ':all:' --default-index 'https://pypi.org/simple' -r "profiles/$ProfileName.txt"
    if ($LASTEXITCODE) {throw 'Runtime dependency installation failed.'}
    & $uv --no-config pip install --python $python --no-index --no-deps $manifest.wheel
    if ($LASTEXITCODE) {throw 'LayerDelta wheel installation failed.'}
    & $uv --no-config pip check --python $python
    if ($LASTEXITCODE) {throw 'Runtime dependency check failed.'}
    & $python -I -c "import torch; print('PyTorch', torch.__version__, '| CPU ready | CUDA available:', torch.cuda.is_available())"
    if ($LASTEXITCODE) {throw 'Runtime import failed.'}
    [IO.File]::WriteAllText((Join-Path $PSScriptRoot $marker), $ProfileName)
    $linkDir = if ($DesktopShortcut) { [Environment]::GetFolderPath('Desktop') } else { $PSScriptRoot }
    $shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $linkDir "LayerDelta-$ProfileName.lnk"))
    $shortcut.TargetPath = Join-Path $PSScriptRoot 'LayerDelta.exe'
    $shortcut.WorkingDirectory = $PSScriptRoot
    $shortcut.Save()
    Write-Host "Open LayerDelta-$ProfileName from $linkDir, or run runtime\Scripts\layerdelta.exe --help."
} finally {
    Pop-Location
}
