[CmdletBinding()]
param([ValidateRange(0,900)][int]$WaitSeconds = 180)

. (Join-Path $PSScriptRoot 'common.ps1')
New-Item -ItemType Directory -Path $script:RuntimeRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath $script:PythonPath -PathType Leaf)) {
    throw 'The .inference-gpu-venv Python environment is missing.'
}
if (-not (Test-Path -LiteralPath $script:ServiceConfig -PathType Leaf)) {
    throw 'The owned local GPU service configuration is missing.'
}

$taskExisting = Get-OwnedInferenceProcess
if ($null -ne $taskExisting) {
    $taskHttp = Get-LocalInferenceHttpStatus -ExpectedProcessId $taskExisting.ProcessId
    [pscustomobject]@{state='already_running';pid=$taskExisting.ProcessId;ready=$taskHttp.Ready;url='http://127.0.0.1:8000'} | ConvertTo-Json -Compress
    return
}
if (@(Get-NetTCPConnection -State Listen -LocalPort 8000 -ErrorAction SilentlyContinue).Count -gt 0) {
    throw 'Port 8000 is already owned by another process. No process was stopped.'
}

$taskPreflight = & $script:PythonPath -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 3)' 2>$null
if ($LASTEXITCODE -ne 0) {
    throw 'The intended inference environment cannot use CUDA yet.'
}
$taskInstance = [guid]::NewGuid().ToString()
$taskEnvNames = @('RETINA_API_KEY','RETINA_SERVICE_CONFIG','RETINA_REGISTRY','RETINA_WEIGHTS_ROOT','RETINA_DEVICE','RETINA_PRECISION','PYTHONUTF8','PYTHONUNBUFFERED')
$taskPreviousEnvironment = @{}
foreach ($taskName in $taskEnvNames) {
    $taskPreviousEnvironment[$taskName] = [Environment]::GetEnvironmentVariable($taskName, 'Process')
}
try {
    [Environment]::SetEnvironmentVariable('RETINA_API_KEY', (Get-RetinaCredential), 'Process')
    [Environment]::SetEnvironmentVariable('RETINA_SERVICE_CONFIG', $script:ServiceConfig, 'Process')
    [Environment]::SetEnvironmentVariable('RETINA_REGISTRY', $null, 'Process')
    [Environment]::SetEnvironmentVariable('RETINA_WEIGHTS_ROOT', $null, 'Process')
    [Environment]::SetEnvironmentVariable('RETINA_DEVICE', 'cuda', 'Process')
    [Environment]::SetEnvironmentVariable('RETINA_PRECISION', 'fp16', 'Process')
    [Environment]::SetEnvironmentVariable('PYTHONUTF8', '1', 'Process')
    [Environment]::SetEnvironmentVariable('PYTHONUNBUFFERED', '1', 'Process')
    $taskArguments = @(('"' + $script:LauncherPath + '"'), '--instance-id', $taskInstance)
    $taskLaunched = Start-Process -FilePath $script:PythonPath -ArgumentList $taskArguments -WorkingDirectory $script:ProjectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $script:RuntimeRoot 'server.stdout.log') -RedirectStandardError (Join-Path $script:RuntimeRoot 'server.stderr.log')
} finally {
    foreach ($taskName in $taskEnvNames) {
        [Environment]::SetEnvironmentVariable($taskName, $taskPreviousEnvironment[$taskName], 'Process')
    }
}

$taskDeadline = [datetime]::UtcNow.AddSeconds($WaitSeconds)
$taskStartedWaiting = [datetime]::UtcNow
$taskOwned = $null
$taskReady = $false
do {
    if (Test-Path -LiteralPath $script:RecordPath -PathType Leaf) {
        $taskCandidate = Get-OwnedInferenceProcess
        if ($null -ne $taskCandidate -and $taskCandidate.InstanceId -eq $taskInstance) {
            $taskOwned = $taskCandidate
            $taskHttp = Get-LocalInferenceHttpStatus -ExpectedProcessId $taskOwned.ProcessId
            if ($taskHttp.Ready) { $taskReady = $true; break }
        }
    }
    if ([datetime]::UtcNow -ge $taskDeadline) { break }
    $taskLaunched.Refresh()
    if ($taskLaunched.HasExited -and $null -eq $taskOwned -and
        ([datetime]::UtcNow - $taskStartedWaiting).TotalSeconds -gt 5) { break }
    Start-Sleep -Milliseconds 1000
} while ($true)

$taskOutputPid = if ($null -ne $taskOwned) {$taskOwned.ProcessId} else {$null}
$taskState = if ($taskReady) {'ready'} elseif ($null -ne $taskOwned) {'starting_or_not_ready'} elseif ($WaitSeconds -eq 0) {'launch_requested'} else {'exited_or_not_started'}
[pscustomobject]@{state=$taskState;pid=$taskOutputPid;launcher_pid=$taskLaunched.Id;instance_id=$taskInstance;ready=$taskReady;url='http://127.0.0.1:8000';stderr_log=(Join-Path $script:RuntimeRoot 'server.stderr.log')} | ConvertTo-Json -Compress
if (-not $taskReady -and $WaitSeconds -gt 0) {
    throw 'Local GPU readiness was not confirmed. Check the owned status and startup log; no cloud service was changed.'
}
