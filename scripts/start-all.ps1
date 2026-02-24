Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "common.ps1")

$config = Get-Config
Start-AllServices -Config $config
Status-AllServices -Config $config