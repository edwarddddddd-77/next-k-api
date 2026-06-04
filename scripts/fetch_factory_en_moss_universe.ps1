# Moss1 目录 -> factory-en data_cache（供 Moss2 回测）
# 在 PowerShell 中于 next-k-api 目录执行:
#   .\scripts\fetch_factory_en_moss_universe.ps1
#   .\scripts\fetch_factory_en_moss_universe.ps1 -DryRun
#   .\scripts\fetch_factory_en_moss_universe.ps1 -SkipExisting
#   .\scripts\fetch_factory_en_moss_universe.ps1 -Bases "BTC,ETH,SOL"

param(
    [switch]$DryRun,
    [switch]$SkipExisting,
    [string]$Bases = "",
    [double]$Sleep = 1.5
)

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$pyArgs = @("scripts/fetch_factory_en_moss_universe.py")
if ($DryRun) { $pyArgs += "--dry-run" }
if ($SkipExisting) { $pyArgs += "--skip-existing" }
if ($Bases) { $pyArgs += @("--bases", $Bases) }
if ($Sleep -ge 0) { $pyArgs += @("--sleep", [string]$Sleep) }

python @pyArgs
exit $LASTEXITCODE
