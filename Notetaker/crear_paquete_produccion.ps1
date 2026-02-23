# Script para crear paquete ZIP de produccion de Notetaker2.0
# Excluye archivos innecesarios como node_modules, venv, .env, etc.

$ErrorActionPreference = "Stop"

# Directorio base del proyecto
$proyectoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$nombreZip = "notetaker2.0-produccion-$(Get-Date -Format 'yyyyMMdd-HHmmss').zip"
$rutaZip = Join-Path $proyectoRoot $nombreZip

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Creando paquete de produccion" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Patrones a excluir
$excluirPatrones = @(
    "node_modules",
    "notetaker",  # Entorno virtual Python
    ".env",
    ".env.*",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".git",
    ".gitignore",
    "logs",
    "*.log",
    "dist",  # Se generara en produccion
    ".vscode",
    ".idea",
    "*.swp",
    "*.swo",
    "*~",
    ".DS_Store",
    "Thumbs.db",
    "Desktop.ini",
    "*.tmp",
    "*.bak",
    "*.cache",
    "celerybeat-schedule*",
    "*.db",
    "*.sqlite",
    "*.sql",
    "connectivity_log.txt",
    "*.dump",
    "postman_*.json",
    "test_*.py",
    "check_*.py",
    "convert.py",
    "add_raw_transcript_json.py",
    "start_ngrok*.bat",
    "verificar_*.ps1",
    "diagnostico_*.ps1",
    "abrir_puertos_*.ps1",
    "sync_ssl_certs.ps1",
    "actualizar_ssl.ps1"
)

# Archivos y carpetas a incluir siempre (aunque coincidan con patrones)
$incluirSiempre = @(
    "env.example",
    "requirements.txt",
    "package.json",
    "README.md",
    "docs",
    "*.md"
)

Write-Host "Buscando archivos a incluir..." -ForegroundColor Yellow

# Obtener todos los archivos del proyecto
$archivos = Get-ChildItem -Path $proyectoRoot -Recurse -File | Where-Object {
    $archivo = $_
    $rutaRelativa = $archivo.FullName.Substring($proyectoRoot.Length + 1)
    
    # Verificar si debe excluirse
    $excluir = $false
    foreach ($patron in $excluirPatrones) {
        if ($rutaRelativa -like "*\$patron\*" -or $rutaRelativa -like "$patron\*" -or $rutaRelativa -eq $patron) {
            $excluir = $true
            break
        }
        # Verificar patrones con wildcards
        if ($patron -like "*.*") {
            $nombreArchivo = Split-Path -Leaf $rutaRelativa
            if ($nombreArchivo -like $patron) {
                $excluir = $true
                break
            }
        }
    }
    
    # Verificar si debe incluirse siempre
    if ($excluir) {
        foreach ($incluir in $incluirSiempre) {
            if ($rutaRelativa -like "*\$incluir" -or $rutaRelativa -eq $incluir -or $rutaRelativa -like $incluir) {
                $excluir = $false
                break
            }
        }
    }
    
    return -not $excluir
}

Write-Host "Encontrados $($archivos.Count) archivos para incluir" -ForegroundColor Green
Write-Host ""

# Crear carpeta temporal para estructura
$tempDir = Join-Path $env:TEMP "notetaker_temp_$(Get-Date -Format 'yyyyMMddHHmmss')"
if (Test-Path $tempDir) {
    Remove-Item $tempDir -Recurse -Force
}
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

Write-Host "Copiando archivos a carpeta temporal..." -ForegroundColor Yellow

# Copiar archivos manteniendo estructura
$contador = 0
foreach ($archivo in $archivos) {
    $contador++
    $rutaRelativa = $archivo.FullName.Substring($proyectoRoot.Length + 1)
    $destino = Join-Path $tempDir $rutaRelativa
    $destinoDir = Split-Path -Parent $destino
    
    # Crear directorio si no existe
    if (-not (Test-Path $destinoDir)) {
        New-Item -ItemType Directory -Path $destinoDir -Force | Out-Null
    }
    
    # Mostrar progreso cada 100 archivos
    if ($contador % 100 -eq 0) {
        Write-Host "  Copiando archivo $contador de $($archivos.Count)..." -ForegroundColor Gray
    }
    
    try {
        Copy-Item $archivo.FullName -Destination $destino -Force
    } catch {
        Write-Warning "Error al copiar archivo $rutaRelativa : $_"
    }
}

Write-Host "Comprimiendo archivos..." -ForegroundColor Yellow

# Comprimir carpeta temporal
if (Test-Path $rutaZip) {
    Remove-Item $rutaZip -Force
}

Compress-Archive -Path "$tempDir\*" -DestinationPath $rutaZip -CompressionLevel Optimal

# Limpiar carpeta temporal
Remove-Item $tempDir -Recurse -Force

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Paquete creado exitosamente!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Archivo: $nombreZip" -ForegroundColor Cyan
Write-Host "Ubicacion: $proyectoRoot" -ForegroundColor Cyan
Write-Host "Tamaño: $([math]::Round((Get-Item $rutaZip).Length / 1MB, 2)) MB" -ForegroundColor Cyan
Write-Host ""
Write-Host "El archivo esta listo para desplegar en produccion." -ForegroundColor Yellow
Write-Host ""
