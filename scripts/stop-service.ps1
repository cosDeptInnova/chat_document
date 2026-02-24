param([Parameter(Mandatory=$true)][string]$Name)

. "$PSScriptRoot\common.ps1"
$config = Get-Config

$svc = $config.Services | Where-Object { $_.Name -eq $Name }
if (-not $svc) { throw "Servicio no encontrado: $Name" }

Stop-ServiceProcess -Config $config -Svc $svc