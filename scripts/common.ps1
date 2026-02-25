Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# =========================
# Core paths / config
# =========================
function Get-RepoRoot {
  (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Ensure-Dirs {
  param([Parameter(Mandatory=$true)][string]$Root)
  foreach ($d in @("run","logs","EVIDENCIAS_HITO2","config","scripts")) {
    $p = Join-Path $Root $d
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p | Out-Null }
  }
}

function Get-Config {
  $cfgPath = Join-Path $PSScriptRoot "services.psd1"
  if (-not (Test-Path $cfgPath)) { throw "No existe config: $cfgPath" }
  Import-PowerShellDataFile -Path $cfgPath
}

# =========================
# StrictMode-proof helpers
# =========================
function As-Array {
  param($x)
  if ($null -eq $x) { return @() }
  if ($x -is [System.Array]) { return $x }
  return @($x)
}

function Count-Of {
  param($x)
  return @($x).Count
}

function Try-GetProp {
  param(
    [Parameter(Mandatory=$true)] $Obj,
    [Parameter(Mandatory=$true)] [string] $Name,
    $Default = $null
  )
  if ($null -eq $Obj) { return $Default }

  $p = $Obj.PSObject.Properties[$Name]
  if ($p) { return $p.Value }

  if ($Obj -is [hashtable] -and $Obj.ContainsKey($Name)) { return $Obj[$Name] }

  return $Default
}

function Try-GetInt {
  param($Obj,[string]$Name,[int]$Default)
  $v = Try-GetProp -Obj $Obj -Name $Name -Default $null
  if ($null -eq $v -or [string]::IsNullOrWhiteSpace("$v")) { return $Default }
  try { return [int]$v } catch { return $Default }
}

function Try-GetBool {
  param($Obj,[string]$Name,[bool]$Default)
  $v = Try-GetProp -Obj $Obj -Name $Name -Default $null
  if ($null -eq $v -or [string]::IsNullOrWhiteSpace("$v")) { return $Default }
  try { return [bool]$v } catch { return $Default }
}

# =========================
# Admin / diagnostics
# =========================
function Test-IsAdmin {
  try {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $p  = New-Object Security.Principal.WindowsPrincipal($id)
    return $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
  } catch {
    return $false
  }
}

function Warn-IfNotAdmin {
  if (-not (Test-IsAdmin)) {
    Write-Host "WARNING: Not running as Administrator. Killing some processes/services may fail."
  }
}

# =========================
# PID / logs
# =========================
function Get-PidFile {
  param([Parameter(Mandatory=$true)][string]$Root,[Parameter(Mandatory=$true)][string]$ServiceName)
  Join-Path (Join-Path $Root "run") "$ServiceName.pid"
}

function Get-LogFiles {
  param([Parameter(Mandatory=$true)][string]$Root,[Parameter(Mandatory=$true)][string]$ServiceName)
  @{
    Out = Join-Path (Join-Path $Root "logs") "$ServiceName.out.log"
    Err = Join-Path (Join-Path $Root "logs") "$ServiceName.err.log"
  }
}

function Read-PidFileSafe {
  param([Parameter(Mandatory=$true)][string]$PidFile)

  if (-not (Test-Path $PidFile)) { return $null }
  $raw = Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $raw) { return $null }

  $raw = ($raw.ToString()).Trim()
  if (-not $raw) { return $null }

  $val = 0
  if (-not [int]::TryParse($raw, [ref]$val)) { return $null }
  if ($val -le 0) { return $null }
  return $val
}

function Is-Running {
  param([Parameter(Mandatory=$true)][string]$PidFile)
  $p = Read-PidFileSafe -PidFile $PidFile
  if ($null -eq $p) { return $false }
  try { Get-Process -Id $p -ErrorAction Stop | Out-Null; return $true } catch { return $false }
}

# =========================
# Ports / owner discovery (robust)
# =========================
function Port-IsListening {
  param([Parameter(Mandatory=$true)][int]$Port)
  $c = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
  return [bool]$c
}

function Get-ListenerPidsByPort {
  param([Parameter(Mandatory=$true)][int]$Port)

  # Primary: Get-NetTCPConnection
  $pids = @(
    Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
      Select-Object -ExpandProperty OwningProcess -Unique |
      Where-Object { $_ -and $_ -gt 0 }
  )

  if ((Count-Of $pids) -gt 0) { return $pids }

  # Fallback: netstat -ano (in case cmdlets behave weird)
  try {
    $lines = @(netstat -ano -p tcp | Select-String -Pattern "LISTENING" | Select-Object -ExpandProperty Line)
    foreach ($ln in $lines) {
      # Example:
      #  TCP    0.0.0.0:7100   0.0.0.0:0   LISTENING   1234
      if ($ln -match ":\s*$Port\s+.+LISTENING\s+(\d+)\s*$") {
        $matchPid = [int]$Matches[1]
        if ($matchPid -gt 0) { $pids += $matchPid }
      }
    }
  } catch { }

  return @($pids | Select-Object -Unique)
}

function Get-ProcessInfoSafe {
  param([Parameter(Mandatory=$true)][int]$ProcessId)
  try {
    return Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction Stop
  } catch {
    return $null
  }
}

function Get-ServicesByPid {
  param([Parameter(Mandatory=$true)][int]$ProcessId)
  try {
    return @(
      Get-CimInstance Win32_Service -ErrorAction SilentlyContinue |
        Where-Object { $_.ProcessId -eq $ProcessId }
    )
  } catch {
    return @()
  }
}

function Test-ProcessAlive {
  param([Parameter(Mandatory=$true)][int]$ProcessId)
  if ($ProcessId -le 0) { return $false }
  try { Get-Process -Id $ProcessId -ErrorAction Stop | Out-Null; return $true } catch { return $false }
}

function Stop-WindowsServiceRobust {
  param([Parameter(Mandatory=$true)][string]$ServiceName)

  if (-not $ServiceName) { return }
  try { Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue } catch { }
  try { sc.exe stop $ServiceName *> $null } catch { }
}

function Stop-ProcessTreeRobust {
  param([Parameter(Mandatory=$true)][int]$ProcessId)

  if (-not (Test-ProcessAlive -ProcessId $ProcessId)) { return }

  # Try soft tree kill via CIM first
  try {
    $children = @(
      Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.ParentProcessId -eq $ProcessId } |
        Select-Object -ExpandProperty ProcessId
    )
    foreach ($childId in (As-Array $children)) {
      if ($childId -and $childId -ne $ProcessId) {
        Stop-ProcessTreeRobust -ProcessId $childId
      }
    }
    if (Test-ProcessAlive -ProcessId $ProcessId) {
      try { Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue } catch { }
    }
  } catch { }

  # Hard fallback: taskkill /T /F (kills full tree reliably)
  if (Test-ProcessAlive -ProcessId $ProcessId) {
    try { taskkill /PID $ProcessId /T /F *> $null } catch { }
  }
}

function Ensure-PortFreeRobust {
  param(
    [Parameter(Mandatory=$true)][int]$Port,
    [string]$Context = "",
    [int]$MaxPasses = 12,
    [int]$SleepMs = 300,
    [switch]$StopServices,
    [switch]$KillPortOwnerAnyProcess
  )

  Warn-IfNotAdmin

  $killed = New-Object 'System.Collections.Generic.HashSet[int]'

  for ($pass=1; $pass -le $MaxPasses; $pass++) {

    $pids = @(Get-ListenerPidsByPort -Port $Port)
    if ((Count-Of $pids) -eq 0) { return $true }

    foreach ($listenerPid in $pids) {
      $pidInt = [int]$listenerPid
      if ($pidInt -le 0) { continue }

      # avoid infinite repeats
      if ($killed.Contains($pidInt)) { continue }

      $pinfo = Get-ProcessInfoSafe -ProcessId $pidInt
      $pname = if ($pinfo -and $pinfo.Name) { $pinfo.Name } else {
        try { (Get-Process -Id $pidInt -ErrorAction SilentlyContinue).ProcessName } catch { $null }
      }
      if (-not $pname) { $pname = "unknown" }
      $cmd   = if ($pinfo) { $pinfo.CommandLine } else { "" }

      # If StopServices enabled and PID is a service: stop it first (it may respawn otherwise)
      if ($StopServices) {
        $svcs = @(Get-ServicesByPid -ProcessId $pidInt)
        foreach ($svc in $svcs) {
          if ($svc -and $svc.Name) {
            Write-Host "[$Context] pass $pass -> STOP service '$($svc.Name)' (PID $pidInt) holding port $Port"
            Stop-WindowsServiceRobust -ServiceName $svc.Name
          }
        }
      }

      # Kill PID (tree)
      if ($KillPortOwnerAnyProcess) {
        Write-Host "[$Context] pass $pass -> KILL PID $pidInt ($pname) holding port $Port"
        Stop-ProcessTreeRobust -ProcessId $pidInt
        [void]$killed.Add($pidInt)
      } else {
        # If you ever want uvicorn-only logic, you can add it here;
        # current requirement: be robust and free the port.
        Write-Host "[$Context] pass $pass -> KILL PID $pidInt ($pname) holding port $Port"
        Stop-ProcessTreeRobust -ProcessId $pidInt
        [void]$killed.Add($pidInt)
      }
    }

    Start-Sleep -Milliseconds $SleepMs
  }

  # Final check + diagnostic
  $still = @(Get-ListenerPidsByPort -Port $Port)
  if ((Count-Of $still) -eq 0) { return $true }

  Write-Host "[$Context] ERROR: Port $Port still LISTEN after retries. PID(s): $($still -join ', ')"

  foreach ($listenerPid in $still) {
    $pidInt = [int]$listenerPid
    $pinfo = Get-ProcessInfoSafe -ProcessId $pidInt
    if ($pinfo) {
      Write-Host "---- PID $pidInt ----"
      Write-Host ("Name: {0}" -f $pinfo.Name)
      if ($pinfo.CommandLine) { Write-Host ("Cmd : {0}" -f $pinfo.CommandLine) }
    } else {
      Write-Host "---- PID $pidInt ----"
      try {
        $gp = Get-Process -Id $pidInt -ErrorAction SilentlyContinue
        if ($gp) { Write-Host ("Name: {0}" -f $gp.ProcessName) }
      } catch { }
    }

    $svcs = @(Get-ServicesByPid -ProcessId $pidInt)
    foreach ($svc in $svcs) {
      if ($svc -and $svc.Name) {
        Write-Host ("Service: {0}  State={1}  StartMode={2}" -f $svc.Name, $svc.State, $svc.StartMode)
      }
    }
  }

  return $false
}

# =========================
# Env discovery / launcher
# =========================
function Get-LocalDotEnvFiles {
  param([Parameter(Mandatory=$true)][string]$ServicePath)
  $candidates = @(
    (Join-Path $ServicePath ".env"),
    (Join-Path $ServicePath ".env.local"),
    (Join-Path $ServicePath ".env.production"),
    (Join-Path $ServicePath ".env.prod")
  )
  $existing = @()
  foreach ($f in $candidates) { if (Test-Path $f) { $existing += $f } }
  return $existing
}

# =========================
# Conda init/activate
# =========================
function Find-CondaExe {
  $candidates = @(
    $env:CONDA_EXE,
    "$env:USERPROFILE\miniconda3\condabin\conda.bat",
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "$env:USERPROFILE\anaconda3\condabin\conda.bat",
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "$env:ProgramData\miniconda3\condabin\conda.bat",
    "$env:ProgramData\miniconda3\Scripts\conda.exe",
    "$env:ProgramData\anaconda3\condabin\conda.bat",
    "$env:ProgramData\anaconda3\Scripts\conda.exe"
  ) | Where-Object { $_ -and (Test-Path $_) }

  if ((Count-Of $candidates) -gt 0) { return $candidates[0] }
  return $null
}

function Initialize-CondaInThisShell {
  if (Get-Command conda -ErrorAction SilentlyContinue) {
    $hook = (& conda "shell.powershell" "hook") | Out-String
    Invoke-Expression $hook
    return
  }

  $condaExe = Find-CondaExe
  if ($condaExe) {
    $hook = (& $condaExe "shell.powershell" "hook") | Out-String
    Invoke-Expression $hook
    return
  }

  throw "Cannot find 'conda'. Run 'conda init powershell' and restart, or open a shell where conda works."
}

function Activate-CondaEnv {
  param([Parameter(Mandatory=$true)][string]$EnvName)

  $already =
    ($env:CONDA_DEFAULT_ENV -and ($env:CONDA_DEFAULT_ENV -ieq $EnvName)) -or
    ($env:CONDA_PREFIX -and ((Split-Path -Leaf $env:CONDA_PREFIX) -ieq $EnvName))

  if ($already) { return }

  Initialize-CondaInThisShell
  conda activate $EnvName | Out-Null
  if (-not $env:CONDA_PREFIX) { throw "Conda activation failed (CONDA_PREFIX empty)." }
}

function New-LauncherScript {
  param(
    [Parameter(Mandatory=$true)][string]$Root,
    [Parameter(Mandatory=$true)][hashtable]$Config,
    [Parameter(Mandatory=$true)]$Svc
  )

  $svcName    = Try-GetProp -Obj $Svc -Name "Name" -Default ""
  $svcPathRel = Try-GetProp -Obj $Svc -Name "Path" -Default ""
  if (-not $svcName)    { throw "Service missing Name in services.psd1" }
  if (-not $svcPathRel) { throw "Service '$svcName' missing Path in services.psd1" }

  $svcPath = Join-Path $Root $svcPathRel
  $runDir  = Join-Path $Root "run"
  $launcherPath = Join-Path $runDir ("launch_{0}.ps1" -f $svcName)

  $localEnvFiles = Get-LocalDotEnvFiles -ServicePath $svcPath

  $allExport = Join-Path $Root "config\ALL_EXPORT.env"
  $globalFromCfgRel = Try-GetProp -Obj $Config -Name "GlobalEnvFile" -Default ""
  $globalFromCfg = if ($globalFromCfgRel) { Join-Path $Root $globalFromCfgRel } else { "" }

  $globalEnvFiles = @()
  if ($globalFromCfg -and (Test-Path $globalFromCfg)) { $globalEnvFiles += $globalFromCfg }
  if (Test-Path $allExport) { $globalEnvFiles += $allExport }

  $svcEnvRel       = Try-GetProp -Obj $Svc -Name "EnvFile" -Default ""
  $svcEnvCandidate = if ($svcEnvRel) { Join-Path $Root $svcEnvRel } else { "" }
  $svcEnvDefault   = Join-Path $Root ("config\{0}.env" -f $svcName)
  $svcEnv          = if ($svcEnvCandidate -and (Test-Path $svcEnvCandidate)) { $svcEnvCandidate }
                     elseif (Test-Path $svcEnvDefault) { $svcEnvDefault }
                     else { "" }

  $python = if ($env:CONDA_PREFIX) { Join-Path $env:CONDA_PREFIX "python.exe" } else { "" }
  if (-not $python -or -not (Test-Path $python)) { $python = "python" }

  $envFilesLog = Join-Path $runDir ("{0}.envfiles.loaded.txt" -f $svcName)

  $envFiles = @()
  $envFiles += $localEnvFiles
  $envFiles += $globalEnvFiles
  if ($svcEnv) { $envFiles += $svcEnv }

  $envFilesArrayLiteral = "@(" + ((
    $envFiles |
      Where-Object { $_ -and (Test-Path $_) } |
      ForEach-Object { "'" + ($_ -replace "'","''") + "'" }
  ) -join ",") + ")"

  $svcArgs = As-Array (Try-GetProp -Obj $Svc -Name "Args" -Default @())
  $svcArgsLiteral = "@(" + ((
    $svcArgs | ForEach-Object { "'" + ($_ -replace "'","''") + "'" }
  ) -join ",") + ")"

  $content = @"
`$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Import-DotEnvFile([string]`$FilePath) {
  if (-not `$FilePath) { return }
  if (-not (Test-Path `$FilePath)) { return }

  Get-Content `$FilePath | ForEach-Object {
    `$line = `$_.Trim()
    if (-not `$line -or `$line.StartsWith('#')) { return }

    `$idx = `$line.IndexOf('=')
    if (`$idx -lt 1) { return }

    `$key = `$line.Substring(0,`$idx).Trim()
    if (`$key.Length -gt 0 -and [int][char]`$key[0] -eq 65279) { `$key = `$key.Substring(1) }

    `$val = `$line.Substring(`$idx+1).Trim()
    if ((`$val.StartsWith('"') -and `$val.EndsWith('"')) -or (`$val.StartsWith("'") -and `$val.EndsWith("'"))) {
      `$val = `$val.Substring(1, `$val.Length-2)
    }

    Set-Item -Path ("Env:{0}" -f `$key) -Value `$val
  }
}

`$svcPath = '$($svcPath -replace "'","''")'
`$envFiles = $envFilesArrayLiteral
`$envFilesLog = '$($envFilesLog -replace "'","''")'

"`$(Get-Date -Format o)  Service=$svcName" | Out-File `$envFilesLog -Encoding utf8
foreach (`$f in `$envFiles) {
  Import-DotEnvFile `$f
  "LOADED: `$f" | Out-File `$envFilesLog -Append -Encoding utf8
}

Set-Item -Path Env:PYTHONUTF8 -Value "1"
Set-Item -Path Env:PYTHONIOENCODING -Value "utf-8"

Set-Item -Path Env:PYTHONPATH -Value `$svcPath
Set-Location `$svcPath

`$python = '$($python -replace "'","''")'
`$svcArgs = $svcArgsLiteral
`$uvArgs  = @('-m','uvicorn') + `$svcArgs

Write-Host ("RUN => {0} {1}" -f `$python, (`$uvArgs -join ' '))
& `$python @uvArgs
"@

  Set-Content -Path $launcherPath -Value $content -Encoding utf8
  return $launcherPath
}

# =========================
# Start / Stop / Status
# =========================
function Start-ServiceProcess {
  param([Parameter(Mandatory=$true)][hashtable]$Config,[Parameter(Mandatory=$true)]$Svc)

  $root = Get-RepoRoot
  Ensure-Dirs -Root $root

  $name    = Try-GetProp -Obj $Svc -Name "Name" -Default ""
  $pathRel = Try-GetProp -Obj $Svc -Name "Path" -Default ""
  $port    = Try-GetInt  -Obj $Svc -Name "Port" -Default 0

  if (-not $name)    { throw "Service missing Name" }
  if (-not $pathRel) { throw "Service '$name' missing Path" }

  $pidFile = Get-PidFile -Root $root -ServiceName $name

  # Clean dead pidfile
  if ((Test-Path $pidFile) -and (-not (Is-Running -PidFile $pidFile))) {
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
  }

  # If port is defined, ALWAYS ensure it is free (robust, no assumptions)
  if ($port -gt 0) {
    $okFree = Ensure-PortFreeRobust -Port $port -Context $name -StopServices -KillPortOwnerAnyProcess
    if (-not $okFree) {
      throw "[$name] Cannot start: port $port is still busy."
    }
  }

  $launcher = New-LauncherScript -Root $root -Config $Config -Svc $Svc
  $logs     = Get-LogFiles -Root $root -ServiceName $name

  Write-Host "[$name] START => powershell -File $launcher"
  $proc = Start-Process -FilePath "powershell.exe" `
                        -ArgumentList @("-NoProfile","-ExecutionPolicy","Bypass","-File",$launcher) `
                        -WorkingDirectory (Join-Path $root $pathRel) `
                        -RedirectStandardOutput $logs.Out `
                        -RedirectStandardError $logs.Err `
                        -WindowStyle Minimized `
                        -PassThru

  Set-Content -Path $pidFile -Value $proc.Id

  $timeout  = Try-GetInt  -Obj $Svc    -Name "StartupTimeoutSec" -Default 180
  $failFast = Try-GetBool -Obj $Config -Name "FailFast"          -Default $false

  if ($port -gt 0) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $timeout) {
      if (Port-IsListening -Port $port) { break }
      Start-Sleep -Milliseconds 300
    }

    if (-not (Port-IsListening -Port $port)) {
      Write-Host "[$name] ERROR: port $port did not open (timeout=${timeout}s). See logs: $($logs.Err)"
      if (Test-Path $logs.Err) { Get-Content $logs.Err -Tail 200 }

      # cleanup best effort
      try { Stop-ProcessTreeRobust -ProcessId $proc.Id } catch { }
      Remove-Item $pidFile -Force -ErrorAction SilentlyContinue

      if ($failFast) { throw "Service ${name} failed to start (port $port not listening)." }
      Write-Host "[$name] Continuing (FailFast=false)."
      return
    }
  }

  Write-Host "[$name] OK (WRAPPER PID $($proc.Id)). Logs: $($logs.Out)"
}

function Stop-ServiceProcess {
  param([Parameter(Mandatory=$true)][hashtable]$Config,[Parameter(Mandatory=$true)]$Svc)

  $root = Get-RepoRoot
  Ensure-Dirs -Root $root

  $name = Try-GetProp -Obj $Svc -Name "Name" -Default ""
  $port = Try-GetInt  -Obj $Svc -Name "Port" -Default 0
  if (-not $name) { return }

  $pidFile = Get-PidFile -Root $root -ServiceName $name

  # 1) Free port (this kills whatever is holding it, including services, with retries)
  if ($port -gt 0) {
    [void](Ensure-PortFreeRobust -Port $port -Context $name -StopServices -KillPortOwnerAnyProcess)
  }

  # 2) Kill wrapper PID if present
  $wrapperProc = Read-PidFileSafe -PidFile $pidFile
  if ($wrapperProc) {
    if (Test-ProcessAlive -ProcessId $wrapperProc) {
      try {
        Write-Host "[$name] STOP wrapper PID $wrapperProc"
        Stop-ProcessTreeRobust -ProcessId $wrapperProc
      } catch { }
    } else {
      Write-Host "[$name] wrapper PID $wrapperProc ya no existe."
    }
  }

  if (Test-Path $pidFile) { Remove-Item $pidFile -Force -ErrorAction SilentlyContinue }

  # 3) Final guarantee: port must be down
  if ($port -gt 0) {
    $still = Port-IsListening -Port $port
    if ($still) {
      Write-Host "[$name] WARNING: port $port still LISTEN after stop. Running final ensure..."
      [void](Ensure-PortFreeRobust -Port $port -Context $name -StopServices -KillPortOwnerAnyProcess -MaxPasses 8)
    }
  }

  Write-Host "[$name] stopped."
}

function Get-ServiceStatus {
  param([Parameter(Mandatory=$true)][hashtable]$Config,[Parameter(Mandatory=$true)]$Svc)

  $root = Get-RepoRoot
  $name = Try-GetProp -Obj $Svc -Name "Name" -Default ""
  $port = Try-GetInt -Obj $Svc -Name "Port" -Default 0
  if (-not $name) { return }

  $pidFile  = Get-PidFile -Root $root -ServiceName $name
  $running  = Is-Running -PidFile $pidFile
  $svcPid   = if ($running) { Read-PidFileSafe -PidFile $pidFile } else { $null }

  [pscustomobject]@{
    Service = $name
    Port    = if ($port -gt 0) { $port } else { "" }
    Running = $running
    PID     = if ($svcPid) { $svcPid } else { "" }
  }
}

# =========================
# Utilities: start/stop all
# =========================
function Get-ServicesFromConfig {
  param([Parameter(Mandatory=$true)][hashtable]$Config)
  $svcs = As-Array (Try-GetProp -Obj $Config -Name "Services" -Default $null)
  if ((Count-Of $svcs) -eq 0) { throw "Invalid config: missing 'Services' in services.psd1" }
  return $svcs
}

function Resolve-StartOrder {
  param([Parameter(Mandatory=$true)][hashtable]$Config)
  return @(As-Array (Try-GetProp -Obj $Config -Name "StartOrder" -Default $null))
}

function Resolve-ServiceByName {
  param(
    [Parameter(Mandatory=$true)][hashtable]$Config,
    [Parameter(Mandatory=$true)][string]$Name
  )

  $target = ($Name | ForEach-Object { $_.Trim() })
  if (-not $target) { throw "Debe indicar un nombre de servicio." }

  $services = @(Get-ServicesFromConfig -Config $Config)

  $exact = @($services | Where-Object {
    $n = Try-GetProp -Obj $_ -Name "Name" -Default ""
    $n -and ($n -ieq $target)
  })

  if ((Count-Of $exact) -eq 1) { return $exact[0] }

  if ((Count-Of $exact) -gt 1) {
    throw "Configuración inválida: existen varios servicios con el nombre '$target'."
  }

  $partial = @($services | Where-Object {
    $n = Try-GetProp -Obj $_ -Name "Name" -Default ""
    $n -and $n.ToLowerInvariant().Contains($target.ToLowerInvariant())
  })

  if ((Count-Of $partial) -eq 1) { return $partial[0] }

  $available = @($services | ForEach-Object { Try-GetProp -Obj $_ -Name "Name" -Default "" } | Where-Object { $_ })

  if ((Count-Of $partial) -gt 1) {
    $matches = ($partial | ForEach-Object { Try-GetProp -Obj $_ -Name "Name" -Default "" }) -join ", "
    throw "Nombre ambiguo '$target'. Coincidencias: $matches. Servicios disponibles: $($available -join ', ')"
  }

  throw "Servicio no encontrado: '$target'. Disponibles: $($available -join ', ')"
}

function Sort-ServicesByStartOrder {
  param(
    [Parameter(Mandatory=$true)]$Services,
    [Parameter(Mandatory=$true)]$StartOrder
  )

  $Services   = @(As-Array $Services)
  $StartOrder = @(As-Array $StartOrder)

  if ((Count-Of $StartOrder) -eq 0) { return $Services }

  $byName = @{}
  foreach ($s in $Services) {
    $n = Try-GetProp -Obj $s -Name "Name" -Default ""
    if ($n) { $byName[$n] = $s }
  }

  $ordered = New-Object System.Collections.Generic.List[object]
  foreach ($n in $StartOrder) {
    if ($byName.ContainsKey($n)) {
      $ordered.Add($byName[$n])
      $null = $byName.Remove($n)
    }
  }

  foreach ($k in @($byName.Keys)) { $ordered.Add($byName[$k]) }

  return @($ordered.ToArray())
}

function Start-AllServices {
  param([Parameter(Mandatory=$true)][hashtable]$Config)

  $root = Get-RepoRoot
  Ensure-Dirs -Root $root

  $envName = Try-GetProp -Obj $Config -Name "CondaEnv" -Default ""
  if ($envName) { Activate-CondaEnv -EnvName $envName }

  $services   = Get-ServicesFromConfig -Config $Config
  $startOrder = Resolve-StartOrder -Config $Config
  $services   = Sort-ServicesByStartOrder -Services $services -StartOrder $startOrder

  foreach ($svc in $services) {
    try {
      Start-ServiceProcess -Config $Config -Svc $svc
    } catch {
      $failFast = Try-GetBool -Obj $Config -Name "FailFast" -Default $false
      Write-Host "ERROR starting service: $($_.Exception.Message)"
      if ($failFast) { throw }
    }
  }
}

function Stop-AllServices {
  param([Parameter(Mandatory=$true)][hashtable]$Config)

  $root = Get-RepoRoot
  Ensure-Dirs -Root $root

  $services   = Get-ServicesFromConfig -Config $Config
  $startOrder = Resolve-StartOrder -Config $Config
  $services   = Sort-ServicesByStartOrder -Services $services -StartOrder $startOrder

  $services = @($services)
  [array]::Reverse($services)

  foreach ($svc in $services) {
    $svcName = Try-GetProp -Obj $svc -Name "Name" -Default "<unknown>"
    try {
      Stop-ServiceProcess -Config $Config -Svc $svc
    } catch {
      Write-Host "ERROR stopping service [$svcName]: $($_.Exception.Message)"
    }
  }
}

function Status-AllServices {
  param([Parameter(Mandatory=$true)][hashtable]$Config)

  $services = Get-ServicesFromConfig -Config $Config
  $rows = foreach ($svc in $services) { Get-ServiceStatus -Config $Config -Svc $svc }
  $rows | Format-Table -AutoSize
}
