[CmdletBinding()]
param()
. (Join-Path $PSScriptRoot 'common.ps1')
$taskOwned = Get-OwnedInferenceProcess
if ($null -eq $taskOwned) {
    [pscustomobject]@{state='not_running';stopped=$false} | ConvertTo-Json -Compress
    return
}
# Re-verify PID, launcher, instance UUID and creation time immediately before stop.
$taskVerified = Get-OwnedInferenceProcess
if ($null -eq $taskVerified) {
    [pscustomobject]@{state='already_exited';stopped=$false} | ConvertTo-Json -Compress
    return
}
if ($taskVerified.InstanceId -ne $taskOwned.InstanceId -or $taskVerified.CreationUtc -ne $taskOwned.CreationUtc) {
    throw 'The owned process changed during stop; no process was stopped.'
}
Stop-Process -Id $taskVerified.ProcessId -ErrorAction Stop
[pscustomobject]@{state='stopped';pid=$taskVerified.ProcessId;instance_id=$taskVerified.InstanceId;stopped=$true;cloud_changed=$false} | ConvertTo-Json -Compress
