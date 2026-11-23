[CmdletBinding()]
param()
. (Join-Path $PSScriptRoot 'common.ps1')
$taskOwned = Get-OwnedInferenceProcess
$taskHttp = if ($null -ne $taskOwned) {Get-LocalInferenceHttpStatus -ExpectedProcessId $taskOwned.ProcessId} else {Get-LocalInferenceHttpStatus}
$taskState = if ($null -eq $taskOwned) {'not_running'} elseif ($taskHttp.Ready) {'ready'} else {'starting_or_not_ready'}
[pscustomobject]@{
    state=$taskState
    process_owned=($null -ne $taskOwned)
    pid=$(if($null -ne $taskOwned){$taskOwned.ProcessId}else{$null})
    instance_id=$(if($null -ne $taskOwned){$taskOwned.InstanceId}else{$null})
    alive=$taskHttp.Alive
    listener_owned=$taskHttp.ListenerOwned
    ready=($null -ne $taskOwned -and $taskHttp.Ready)
    url='http://127.0.0.1:8000'
} | ConvertTo-Json -Compress
