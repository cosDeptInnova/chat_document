param(
  [string]$RepoRoot = "",            # si lo dejas vacío, se auto-detecta subiendo carpetas hasta docker-compose.yml
  [switch]$Merge = $true,            # merge con config\<svc>.env existente
  [switch]$ImportLocalDotEnv = $true,# lee .env local del microservicio si existe
  [switch]$Show,                     # muestra por consola el contenido resultante
  [switch]$ShowSecrets,              # si lo pones, NO enmascara secretos en pantalla
  [switch]$Backup = $true            # crea backup del config\<svc>.env antes de sobrescribir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RepoRootAuto {
  $p = (Get-Location).Path
  while ($true) {
    if (Test-Path (Join-Path $p "docker-compose.yml")) { return $p }
    $parent = Split-Path -Parent $p
    if (-not $parent -or $parent -eq $p) { break }
    $p = $parent
  }
  throw "No puedo auto-detectar el RepoRoot. Pásalo explícito: -RepoRoot 'C:\...\00.-ASISTENTE_VIRTUAL_COSMOS'"
}

function Parse-DotEnvFile {
  param([string]$Path)
  $dict = @{}
  if (-not (Test-Path $Path)) { return $dict }

  Get-Content $Path | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#")) { return }

    $idx = $line.IndexOf("=")
    if ($idx -lt 1) { return }

    $k = $line.Substring(0,$idx).Trim()
    $v = $line.Substring($idx+1).Trim()

    # strip comillas
    if (($v.StartsWith('"') -and $v.EndsWith('"')) -or ($v.StartsWith("'") -and $v.EndsWith("'"))) {
      $v = $v.Substring(1, $v.Length-2)
    }
    $dict[$k] = $v
  }
  return $dict
}

function Write-DotEnvFile {
  param([string]$Path,[hashtable]$Vars,[switch]$Backup)

  if ($Backup -and (Test-Path $Path)) {
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    Copy-Item $Path "$Path.bak_$ts" -Force
  }

  $lines = New-Object System.Collections.Generic.List[string]
  foreach ($k in ($Vars.Keys | Sort-Object)) {
    $v = [string]$Vars[$k]
    if ($null -eq $v) { $v = "" }

    # Si contiene espacios o caracteres “peligrosos”, lo comillamos
    if ($v -match '\s' -or $v -match '[#"'']') {
      $vEsc = $v.Replace('"','\"')
      $lines.Add("$k=`"$vEsc`"")
    } else {
      $lines.Add("$k=$v")
    }
  }
  $lines | Out-File -FilePath $Path -Encoding utf8
}

function Mask {
  param([string]$Key,[string]$Val,[switch]$ShowSecrets)
  if ($ShowSecrets) { return $Val }
  if ($Key -match '(?i)secret|token|key|password|pwd') {
    if (-not $Val) { return "" }
    return "<SET len=$($Val.Length)>"
  }
  return $Val
}

function Extract-EnvKeysFromSessionHistory {
  # Solo coge nombres de variables que hayas seteado con $env:VAR = ...
  $keys = New-Object System.Collections.Generic.HashSet[string]

  $hist = @()
  try { $hist = Get-History | Select-Object -ExpandProperty CommandLine } catch { $hist = @() }

  # Patrón de asignación tipo: $env:FOO="bar" / $env:FOO = bar / $env:FOO='bar'
  $re = '^\s*\$env:([A-Za-z_][A-Za-z0-9_]*)\s*='

  foreach ($line in $hist) {
    $m = [regex]::Match($line, $re)
    if ($m.Success) { [void]$keys.Add($m.Groups[1].Value) }
  }

  return @($keys)
}

# ===== MAIN =====
if (-not $RepoRoot) { $RepoRoot = Resolve-RepoRootAuto } else { $RepoRoot = (Resolve-Path $RepoRoot).Path }

$svcName = Split-Path -Leaf (Get-Location)
$configDir = Join-Path $RepoRoot "config"
New-Item -ItemType Directory -Force $configDir | Out-Null

$outFile = Join-Path $configDir ("{0}.env" -f $svcName)

# 1) Base: lo que ya exista en config\<svc>.env
$merged = @{}
if ($Merge -and (Test-Path $outFile)) {
  $base = Parse-DotEnvFile -Path $outFile
  foreach ($k in $base.Keys) { $merged[$k] = $base[$k] }
}

# 2) Importar .env local del microservicio (si existe) para compatibilidad
if ($ImportLocalDotEnv) {
  $localEnvPath = Join-Path (Get-Location).Path ".env"
  if (Test-Path $localEnvPath) {
    $local = Parse-DotEnvFile -Path $localEnvPath
    foreach ($k in $local.Keys) {
      if (-not $merged.ContainsKey($k)) { $merged[$k] = $local[$k] }
    }
  }
}

# 3) Variables seteadas con $env: en ESTA sesión (Get-History)
$keys = Extract-EnvKeysFromSessionHistory
if ($keys.Count -eq 0) {
  Write-Host "AVISO: No encuentro asignaciones '$env:VAR=' en el historial de esta consola (Get-History)."
  Write-Host "       Si las pusiste en otra consola, ejecuta este script allí."
}

foreach ($k in $keys) {
  $v = (Get-Item ("Env:{0}" -f $k) -ErrorAction SilentlyContinue).Value
  if ($null -eq $v -or $v -eq "") {
    # si está vacío, lo dejamos como placeholder para que lo veas
    $merged[$k] = "__MISSING__"
  } else {
    $merged[$k] = $v
  }
}

# 4) Escribir config\<svc>.env
Write-DotEnvFile -Path $outFile -Vars $merged -Backup:$Backup

Write-Host "OK: generado/actualizado -> $outFile"
Write-Host "   Servicio detectado (directorio actual): $svcName"
Write-Host "   Vars detectadas desde Get-History: $($keys.Count)"

if ($Show) {
  Write-Host ""
  Write-Host "===== CONTENIDO ($outFile) ====="
  foreach ($k in ($merged.Keys | Sort-Object)) {
    $mv = Mask -Key $k -Val ([string]$merged[$k]) -ShowSecrets:$ShowSecrets
    "{0}={1}" -f $k, $mv
  }
}