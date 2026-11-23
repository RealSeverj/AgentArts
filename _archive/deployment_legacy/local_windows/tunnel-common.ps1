Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:TunnelProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$script:TunnelRuntimeRoot = Join-Path $script:TunnelProjectRoot 'inference_runtime\local_gpu'
$script:TunnelStatePath = Join-Path $script:TunnelRuntimeRoot 'tunnel-state.json'
$script:TunnelExe = Join-Path $script:TunnelRuntimeRoot 'tools\cloudflared-windows-amd64-2026.11.0.exe'
$script:TunnelConfig = Join-Path $PSScriptRoot 'tunnel-empty.yml'
$script:TunnelOrigin = 'http://127.0.0.1:8000'
$script:TunnelExpectedSha256 = 'd5c1d82031da6253dcdcff64c944cd0fe063016edeb2e4fd088075fc9bd52d0a'

function Read-TunnelState {
    if (-not (Test-Path -LiteralPath $script:TunnelStatePath)) { return $null }
    return Get-Content -LiteralPath $script:TunnelStatePath -Raw | ConvertFrom-Json
}

function Write-TunnelState($State) {
    $State | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $script:TunnelStatePath -Encoding UTF8
}

function Get-OwnedTunnelProcess($State) {
    if ($null -eq $State) { return $null }
    $taskProcess = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + [int]$State.pid) -ErrorAction Stop
    if ($null -eq $taskProcess) { return $null }
    $taskCreationUtc = ([datetime]$taskProcess.CreationDate).ToUniversalTime()
    # ConvertFrom-Json can deserialize an ISO timestamp as DateTime on newer
    # PowerShell. Casting that object to string loses its fractional seconds.
    # Compare the exact UTC ticks instead of locale-dependent text formatting.
    try {
        if ($State.created_utc -is [datetimeoffset]) {
            $taskRecordedCreationUtc = $State.created_utc.UtcDateTime
        } elseif ($State.created_utc -is [datetime]) {
            $taskRecordedCreationUtc = $State.created_utc.ToUniversalTime()
        } else {
            $taskRecordedCreationUtc = [datetimeoffset]::Parse(
                [string]$State.created_utc, [Globalization.CultureInfo]::InvariantCulture,
                [Globalization.DateTimeStyles]::AssumeUniversal).UtcDateTime
        }
    } catch {
        throw 'Saved tunnel creation time is invalid; refusing to control the process.'
    }
    if (-not [string]::Equals([string]$taskProcess.ExecutablePath, $script:TunnelExe, [StringComparison]::OrdinalIgnoreCase) -or
        $taskCreationUtc.Ticks -ne $taskRecordedCreationUtc.Ticks -or
        -not ([string]$taskProcess.CommandLine).Contains($script:TunnelOrigin) -or
        -not ([string]$taskProcess.CommandLine).Contains($script:TunnelConfig)) {
        throw 'Saved tunnel PID belongs to a different process; refusing to control it.'
    }
    return $taskProcess
}

function Update-TunnelUrl($State) {
    foreach ($taskLogPath in @($State.stdout_log, $State.stderr_log)) {
        if (-not [string]::IsNullOrWhiteSpace([string]$taskLogPath) -and
            (Test-Path -LiteralPath $taskLogPath -PathType Leaf)) {
            $taskLogText = Get-Content -LiteralPath $taskLogPath -Raw -ErrorAction SilentlyContinue
            # A newly redirected stdout file is often empty while cloudflared
            # writes its startup URL to stderr. Get-Content -Raw then yields
            # null, which must not be passed to Regex.Match under PowerShell.
            if ($null -eq $taskLogText -or [string]::IsNullOrWhiteSpace([string]$taskLogText)) {
                continue
            }
            $taskUrlMatch = [regex]::Match([string]$taskLogText, 'https://[a-z0-9]+(?:-[a-z0-9]+)+\.trycloudflare\.com\b')
            if ($taskUrlMatch.Success) {
                $State.public_url = $taskUrlMatch.Value
                return
            }
        }
    }
}

function Get-TunnelSummary($State, [bool]$Running) {
    if ($null -eq $State) {
        return [ordered]@{ running = $false; public_url = $null; message = 'Tunnel has not been started by these scripts.' }
    }
    return [ordered]@{
        running = $Running
        pid = $State.pid
        public_url = $State.public_url
        local_origin = $script:TunnelOrigin
        created_utc = $State.created_utc
        temporary = $true
        state_file = $script:TunnelStatePath
        stdout_log = $State.stdout_log
        stderr_log = $State.stderr_log
    }
}
