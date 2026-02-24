param([Parameter(Mandatory=$true)][string]$Name)

. "$PSScriptRoot\common.ps1"
$config = Get-Config

Activate-CondaEnv -EnvName $config.CondaEnv

$svc = $config.Services | Where-Object { $_.Name -eq $Name }
if (-not $svc) { throw "Servicio no encontrado: $Name" }

Start-ServiceProcess -Config $config -Svc $svc