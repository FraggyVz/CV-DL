# Run from any directory. Arguments are forwarded to the batch CLI.
$ErrorActionPreference = 'Stop'
$interpreter = Join-Path $PSScriptRoot '.venv-modular\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $interpreter)) {
    $interpreter = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $interpreter)) {
    throw 'Create .venv or .venv-modular and install requirements.txt first; see README.md.'
}
Push-Location $PSScriptRoot
try {
    & $interpreter -m stratscout process-recordings @args
    $resultCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $resultCode
