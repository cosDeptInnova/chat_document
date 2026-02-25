param(
  [Parameter(Mandatory=$true)][string]$Name,
  [switch]$ShowStatus
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\common.ps1"
$config = Get-Config

Activate-CondaEnv -EnvName $config.CondaEnv

$svc = Resolve-ServiceByName -Config $config -Name $Name

Start-ServiceProcess -Config $config -Svc $svc

if ($ShowStatus) {
  Status-AllServices -Config $config
}
