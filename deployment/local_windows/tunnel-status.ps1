[CmdletBinding()]
param()
. (Join-Path $PSScriptRoot 'tunnel-common.ps1')
$taskState = Read-TunnelState
$taskRunning = $null -ne (Get-OwnedTunnelProcess $taskState)
if ($null -ne $taskState -and $taskRunning) {
    Update-TunnelUrl $taskState
    Write-TunnelState $taskState
}
Get-TunnelSummary $taskState $taskRunning | ConvertTo-Json -Depth 4
