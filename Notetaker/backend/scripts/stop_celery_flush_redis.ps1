# Detener Celery, limpiar Redis (en Docker) y opcionalmente reiniciar Celery
# Uso: ejecutar desde backend o desde scripts con -BackendPath si hace falta

param(
    [string]$BackendPath = "",
    [switch]$SkipRestart
)

$ErrorActionPreference = "Stop"

# Resolver ruta del backend (donde esta .env y start_celery.bat)
if ($BackendPath) {
    $backendDir = $BackendPath
} else {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $backendDir = Split-Path -Parent $scriptDir
}
if (-not (Test-Path $backendDir)) {
    Write-Host "Error: directorio backend no encontrado: $backendDir"
    exit 1
}

Write-Host "=== 1. Detener Celery ==="
Write-Host "Deten Celery manualmente (Ctrl+C en la ventana donde corre) o se intentara detener procesos."
$procsToStop = @()
try {
    $procsToStop = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*celery*" } |
        ForEach-Object { $_.ProcessId }
} catch {
    # Sin WMI/CIM, no matar procesos; solo avisar
}
if ($procsToStop.Count -gt 0) {
    foreach ($procId in $procsToStop) {
        try {
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            Write-Host "Detenido PID $procId (celery)"
        } catch {
            Write-Host "No se pudo detener PID $procId : $_"
        }
    }
} else {
    Write-Host "No se encontraron procesos python+celery. Si Celery esta en otra ventana, detenlo con Ctrl+C."
}

Write-Host ""
Write-Host "=== 2. Limpiar Redis (Docker) ==="
# Buscar contenedor Redis: por puerto 6380, 6379 o por nombre
$redisContainer = $null
if (Get-Command docker -ErrorAction SilentlyContinue) {
    $raw = @(docker ps --filter "publish=6380" --format "{{.Names}}" 2>$null | Where-Object { $_.Trim() })
    if (-not $raw) {
        $raw = @(docker ps --filter "publish=6379" --format "{{.Names}}" 2>$null | Where-Object { $_.Trim() })
    }
    if (-not $raw) {
        $raw = @(docker ps --filter "name=redis" --format "{{.Names}}" 2>$null | Where-Object { $_.Trim() })
    }
    # En PowerShell la salida de docker a veces llega como una linea por caracter; unir para nombre completo
    if ($raw.Count -gt 0) {
        if ($raw.Count -gt 1 -and $raw[0].Length -eq 1) {
            $redisContainer = ($raw -join "").Trim()
        } else {
            $redisContainer = $raw[0].Trim()
        }
    }
}

if ($redisContainer) {
    Write-Host "Contenedor Redis: $redisContainer"
    $out = docker exec $redisContainer redis-cli FLUSHDB 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Redis FLUSHDB ejecutado correctamente: $out"
    } else {
        Write-Host "Error al ejecutar FLUSHDB: $out"
    }
} else {
    Write-Host "No se encontro contenedor Redis. Comandos manuales:"
    Write-Host "  docker ps"
    Write-Host "  docker exec <nombre_contenedor_redis> redis-cli FLUSHDB"
}

Write-Host ""
Write-Host "=== 3. Verificar procesos Python/Celery ==="
Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like "*python*" } | ForEach-Object {
    Write-Host "  PID $($_.Id): $($_.ProcessName)"
}

Write-Host ""
if (-not $SkipRestart) {
    Write-Host "=== 4. Reiniciar Celery ==="
    $bat = Join-Path $backendDir "start_celery.bat"
    if (Test-Path $bat) {
        Start-Process cmd -ArgumentList "/c", "cd /d `"$backendDir`" && start_celery.bat" -WorkingDirectory $backendDir
        Write-Host "Celery iniciado en nueva ventana (start_celery.bat)."
    } else {
        Write-Host "No se encontro start_celery.bat en $backendDir. Inicialo manualmente."
    }
} else {
    Write-Host "=== 4. Reinicio omitido (-SkipRestart). Ejecuta start_celery.bat manualmente. ==="
}
