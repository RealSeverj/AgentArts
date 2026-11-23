# Shared helpers. Never print API credentials or signed image URLs.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$script:RuntimeRoot = Join-Path $script:ProjectRoot 'inference_runtime\local_gpu'
$script:PythonPath = Join-Path $script:ProjectRoot '.inference-gpu-venv\Scripts\python.exe'
$script:LauncherPath = Join-Path $PSScriptRoot 'run_server.py'
$script:RecordPath = Join-Path $script:RuntimeRoot 'server_process.json'
$script:ServiceConfig = Join-Path $script:RuntimeRoot 'service.json'
$script:CredentialPath = Join-Path $script:ProjectRoot 'inference_runtime\retina_api_key.dpapi'

function Get-OwnedInferenceProcess {
    if (-not (Test-Path -LiteralPath $script:RecordPath -PathType Leaf)) { return $null }
    $taskRecord = Get-Content -LiteralPath $script:RecordPath -Raw | ConvertFrom-Json
    if ($taskRecord.kind -ne 'retina_local_gpu_inference_process_v1') {
        throw 'The process record is not an owned local inference record.'
    }
    $taskPid = 0
    if (-not [int]::TryParse([string]$taskRecord.pid, [ref]$taskPid) -or $taskPid -lt 1) {
        throw 'The owned process record contains an invalid PID.'
    }
    $taskInstance = [guid]::Empty
    if (-not [guid]::TryParse([string]$taskRecord.instance_id, [ref]$taskInstance)) {
        throw 'The owned process record contains an invalid instance identifier.'
    }
    if (-not [string]::Equals([IO.Path]::GetFullPath([string]$taskRecord.launcher),
                              [IO.Path]::GetFullPath($script:LauncherPath),
                              [StringComparison]::OrdinalIgnoreCase)) {
        throw 'The process record points to a different launcher.'
    }
    $taskProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $taskPid" -ErrorAction Stop
    if ($null -eq $taskProcess) { return $null }
    $taskCommandLine = [string]$taskProcess.CommandLine
    $taskLauncherPattern = '(?i)(?:^|[\s"])' + [regex]::Escape($script:LauncherPath) + '(?:[\s"]|$)'
    $taskInstancePattern = '(?i)(?:^|\s)--instance-id\s+"?' + [regex]::Escape($taskInstance.ToString()) + '(?:"|\s|$)'
    if ($taskCommandLine -notmatch $taskLauncherPattern -or $taskCommandLine -notmatch $taskInstancePattern) {
        throw 'PID ownership could not be verified; refusing to manage that process.'
    }
    $taskExecutable = [IO.Path]::GetFullPath([string]$taskProcess.ExecutablePath)
    if ([IO.Path]::GetFileName($taskExecutable) -notin @('python.exe', 'pythonw.exe')) {
        throw 'The recorded process is not a Python interpreter.'
    }
    # The script path AND random instance UUID above must match. Comparing the
    # creation time also prevents a stale PID record from matching a reused PID.
    $taskRecordedUtc = [DateTimeOffset]::Parse([string]$taskRecord.recorded_utc)
    $taskCreatedUtc = ([datetime]$taskProcess.CreationDate).ToUniversalTime()
    if ([math]::Abs(($taskRecordedUtc.UtcDateTime - $taskCreatedUtc).TotalSeconds) -gt 30) {
        throw 'The process creation time does not match the owned record.'
    }
    [pscustomobject]@{
        ProcessId = $taskPid
        InstanceId = $taskInstance.ToString()
        CreationUtc = $taskCreatedUtc.ToString('o')
        Record = $taskRecord
    }
}

function Get-LocalInferenceHttpStatus {
    param([int]$ExpectedProcessId = 0)
    $taskHealth = $false
    $taskReady = $false
    $taskListenerOwned = $false
    if ($ExpectedProcessId -gt 0) {
        $taskListenerOwned = @(Get-NetTCPConnection -State Listen -LocalAddress '127.0.0.1' -LocalPort 8000 -ErrorAction SilentlyContinue |
            Where-Object {$_.OwningProcess -eq $ExpectedProcessId}).Count -gt 0
        if (-not $taskListenerOwned) {
            return [pscustomobject]@{Alive=$false; Ready=$false; ListenerOwned=$false}
        }
    }
    foreach ($taskPath in @('/healthz', '/readyz')) {
        try {
            $taskResponse = Invoke-RestMethod -Uri ('http://127.0.0.1:8000' + $taskPath) -TimeoutSec 3 -Method Get
            if ($taskPath -eq '/healthz') { $taskHealth = ($taskResponse.alive -eq $true) }
            if ($taskPath -eq '/readyz') { $taskReady = ($taskResponse.ready -eq $true) }
        } catch {
            # Do not echo HTTP exceptions or response data.
        }
    }
    [pscustomobject]@{Alive=$taskHealth; Ready=$taskReady; ListenerOwned=$taskListenerOwned}
}

function Get-RetinaCredential {
    if (-not (Test-Path -LiteralPath $script:CredentialPath -PathType Leaf)) {
        throw 'The current-user DPAPI credential file is missing.'
    }
    Add-Type -AssemblyName System.Security
    $taskEncrypted = [IO.File]::ReadAllBytes($script:CredentialPath)
    $taskPlainBytes = $null
    try {
        $taskPlainBytes = [Security.Cryptography.ProtectedData]::Unprotect(
            $taskEncrypted, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser)
        $taskCredential = [Text.Encoding]::UTF8.GetString($taskPlainBytes)
        if ([Text.Encoding]::UTF8.GetByteCount($taskCredential) -lt 32) {
            throw 'The decrypted credential is invalid.'
        }
        return $taskCredential
    } catch {
        throw 'The API credential could not be read with this Windows user account.'
    } finally {
        if ($null -ne $taskPlainBytes) { [Array]::Clear($taskPlainBytes, 0, $taskPlainBytes.Length) }
        [Array]::Clear($taskEncrypted, 0, $taskEncrypted.Length)
    }
}
