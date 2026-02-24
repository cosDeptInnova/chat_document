. "$PSScriptRoot\common.ps1"
$config = Get-Config

$rows = foreach ($svc in $config.Services) { Get-ServiceStatus -Config $config -Svc $svc }
$rows | Sort-Object Service | Format-Table -AutoSize