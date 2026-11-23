[CmdletBinding()]
param([ValidateRange(5, 60)][int]$WaitSeconds = 45)
. (Join-Path $PSScriptRoot 'common.ps1')
. (Join-Path $PSScriptRoot 'tunnel-common.ps1')

function Get-TunnelOriginOwner {
    # This shared helper checks the fixed record kind/path, launcher path,
    # random instance UUID in the command line, and process creation time.
    # Dot-sourcing common.ps1 only declares helpers; it does not read the key.
    $taskOriginOwner = Get-OwnedInferenceProcess
    if ($null -eq $taskOriginOwner) {
        throw 'The owned local inference process record is missing or its process is not running.'
    }
    foreach ($taskPathCheck in @(
        @([string]$taskOriginOwner.Record.project_root, $script:ProjectRoot),
        @([string]$taskOriginOwner.Record.config, $script:ServiceConfig),
        @([string]$taskOriginOwner.Record.python, $script:PythonPath)
    )) {
        if (-not [string]::Equals([IO.Path]::GetFullPath($taskPathCheck[0]),
                                 [IO.Path]::GetFullPath($taskPathCheck[1]),
                                 [StringComparison]::OrdinalIgnoreCase)) {
            throw 'The inference process record points outside the intended local GPU deployment.'
        }
    }
    if ([string]$taskOriginOwner.Record.listen_url -ne $script:TunnelOrigin) {
        throw 'The inference process record declares a different listener.'
    }
    $taskOriginListeners = @(Get-NetTCPConnection -State Listen -LocalPort 8000 -ErrorAction Stop)
    if ($taskOriginListeners.Count -ne 1 -or
        $taskOriginListeners[0].LocalAddress -ne '127.0.0.1' -or
        $taskOriginListeners[0].OwningProcess -ne $taskOriginOwner.ProcessId) {
        throw 'The loopback listener does not belong to the verified local inference process.'
    }
    return $taskOriginOwner
}

New-Item -ItemType Directory -Path $script:TunnelRuntimeRoot -Force | Out-Null
$taskInferenceOwner = Get-TunnelOriginOwner
$taskExisting = Read-TunnelState
if ($null -ne $taskExisting) {
    $taskExistingProcess = Get-OwnedTunnelProcess $taskExisting
    if ($null -ne $taskExistingProcess) {
        Update-TunnelUrl $taskExisting
        Write-TunnelState $taskExisting
        Get-TunnelSummary $taskExisting $true | ConvertTo-Json -Depth 4
        return
    }
}

if (-not (Test-Path -LiteralPath $script:TunnelExe)) { throw 'Verified cloudflared executable is missing.' }
$taskHash = (Get-FileHash -LiteralPath $script:TunnelExe -Algorithm SHA256).Hash.ToLowerInvariant()
if ($taskHash -ne $script:TunnelExpectedSha256) { throw 'cloudflared SHA256 mismatch; refusing to execute.' }

# Require the owned, ready, authenticated application before exposing its port.
$taskReady = Invoke-RestMethod -Uri ($script:TunnelOrigin + '/readyz') -TimeoutSec 5
if ($taskReady.ready -ne $true) { throw 'Inference service is not ready.' }
$taskUnauthStatus = $null
try {
    $taskUnexpected = Invoke-WebRequest -Uri ($script:TunnelOrigin + '/v1/models') -UseBasicParsing -TimeoutSec 5
    $taskUnauthStatus = [int]$taskUnexpected.StatusCode
} catch {
    if ($null -ne $_.Exception.Response) { $taskUnauthStatus = [int]$_.Exception.Response.StatusCode }
    else { throw 'Unable to verify the application authentication boundary.' }
}
if ($taskUnauthStatus -ne 401) { throw 'Application must reject unauthenticated model access with HTTP401.' }

# Recheck identity after HTTP probes, immediately before preparing the tunnel.
$taskConfirmedOwner = Get-TunnelOriginOwner
if ($taskConfirmedOwner.ProcessId -ne $taskInferenceOwner.ProcessId -or
    $taskConfirmedOwner.InstanceId -ne $taskInferenceOwner.InstanceId -or
    $taskConfirmedOwner.CreationUtc -ne $taskInferenceOwner.CreationUtc) {
    throw 'The local inference process changed during tunnel preflight.'
}

$taskLogRoot = Join-Path $script:TunnelRuntimeRoot 'tunnel_logs'
New-Item -ItemType Directory -Path $taskLogRoot -Force | Out-Null
$taskRunId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$taskStdout = Join-Path $taskLogRoot ($taskRunId + '.stdout.log')
$taskStderr = Join-Path $taskLogRoot ($taskRunId + '.stderr.log')
# An explicit empty configuration avoids consuming the user's named-tunnel config.
# INFO logs do not emit request headers/body; never change this to DEBUG.
$taskArguments = @('tunnel', '--config', ('"' + $script:TunnelConfig + '"'), '--url', $script:TunnelOrigin,
    '--protocol', 'http2', '--edge-ip-version', '4', '--metrics', '127.0.0.1:0', '--loglevel', 'info', '--management-diagnostics=false', '--no-autoupdate')
# cloudflared does not need the inference key, MaaS key, or any named-tunnel credential.
$taskRemovedEnvironment = @{}
foreach ($taskEnvName in @([Environment]::GetEnvironmentVariables('Process').Keys)) {
    if ([string]$taskEnvName -match '^(RETINA_API_KEY|MAAS_API_KEY|TUNNEL_.*)$') {
        $taskRemovedEnvironment[[string]$taskEnvName] = [Environment]::GetEnvironmentVariable([string]$taskEnvName, 'Process')
        [Environment]::SetEnvironmentVariable([string]$taskEnvName, $null, 'Process')
    }
}
try {
    $taskProcess = Start-Process -FilePath $script:TunnelExe -ArgumentList $taskArguments -WorkingDirectory $script:TunnelProjectRoot -WindowStyle Hidden -RedirectStandardOutput $taskStdout -RedirectStandardError $taskStderr -PassThru
} finally {
    foreach ($taskEnvName in $taskRemovedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($taskEnvName, $taskRemovedEnvironment[$taskEnvName], 'Process')
    }
    $taskRemovedEnvironment.Clear()
}
$taskOwnedProcess = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $taskProcess.Id)
if ($null -eq $taskOwnedProcess) { throw 'cloudflared exited immediately; inspect the INFO startup log.' }
$taskState = [pscustomobject][ordered]@{
    pid = $taskProcess.Id
    created_utc = $taskOwnedProcess.CreationDate.ToUniversalTime().ToString('o')
    executable = $script:TunnelExe
    executable_sha256 = $taskHash
    local_origin = $script:TunnelOrigin
    origin_pid = $taskInferenceOwner.ProcessId
    origin_instance_id = $taskInferenceOwner.InstanceId
    origin_created_utc = $taskInferenceOwner.CreationUtc
    origin_record = $script:RecordPath
    public_url = $null
    stdout_log = $taskStdout
    stderr_log = $taskStderr
    stopped_utc = $null
}
Write-TunnelState $taskState
$taskDeadline = [DateTime]::UtcNow.AddSeconds($WaitSeconds)
while ([DateTime]::UtcNow -lt $taskDeadline) {
    $taskProcess.Refresh()
    if ($taskProcess.HasExited) { throw 'cloudflared exited; inspect the INFO startup log. The inference server remains running.' }
    Update-TunnelUrl $taskState
    if ($null -ne $taskState.public_url) { break }
    Start-Sleep -Milliseconds 500
}
Write-TunnelState $taskState
Get-TunnelSummary $taskState $true | ConvertTo-Json -Depth 4
if ($null -eq $taskState.public_url) {
    Write-Warning 'Tunnel process is still starting; run tunnel-status.ps1. Do not change the AgentArts plugin until external HTTPS and authentication tests pass.'
}
