# Usage: .\run_all.ps1 [-Seeds 1,7,42] [-Types "DISEASE"]
param(
  [int[]]$Seeds = @(1, 7, 42),
  [string[]]$Types = @("DISEASE"),
  [string[]]$Models = @("bilstm", "bilstm_attn", "cnn_bilstm", "cnn_bilstm_attn", "cnn_bigru_attn"),
  [string]$PythonExe = "C:\Users\manis\.gemini\antigravity\scratch\v\Scripts\python.exe"
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $PythonExe)) { $PythonExe = "python" }
$srcDir = Join-Path $PSScriptRoot "src"
Set-Location $srcDir

foreach ($T in $Types) {
  $t = $T.ToLower()
  & $PythonExe prepare_data.py --data_root ../data --corpus spaccc --types $T --out ../processed_$t
  foreach ($M in $Models) {
    foreach ($S in $Seeds) {
      Write-Host "=== Running $M on $t (Seed $S) ===" -ForegroundColor Cyan
      & $PythonExe train.py    --data ../processed_$t --model $M --seed $S --out ../runs_$t
      & $PythonExe evaluate.py --data ../processed_$t --run ../runs_$t/${M}_seed$S
    }
  }
  & $PythonExe summarize.py ../runs_$t
}
& $PythonExe make_report.py ..
