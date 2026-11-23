[CmdletBinding()]
param()
. (Join-Path $PSScriptRoot 'tunnel-common.ps1')
$taskState = Read-TunnelState
$taskOwnedProcess = Get-OwnedTunnelProcess $taskState
if ($null -ne $taskOwnedProcess) {
    # Executable path, creation time and the exact local origin/config were checked.
    Stop-Process -Id ([int]$taskState.pid) -ErrorAction Stop
}
if ($null -ne $taskState) {
    $taskState.stopped_utc = [DateTime]::UtcNow.ToString('o')
    Write-TunnelState $taskState
}
Get-TunnelSummary $taskState $false | ConvertTo-Json -Depth 4
