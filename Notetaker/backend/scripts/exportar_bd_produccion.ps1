# Script para EXPORTAR BD de produccion (ejecutar en el servidor de produccion)
# Uso: .\scripts\exportar_bd_produccion.ps1

param(
    [Parameter(Mandatory=$false)]
    [string]$DatabaseUrl = "postgresql://cosmos_user:Cos4321@localhost:5432/cosmos_notetaker",
    
    [Parameter(Mandatory=$false)]
    [string]$BackupFile = "backup_produccion_$(Get-Date -Format 'yyyyMMdd_HHmmss').sql",
    
    [Parameter(Mandatory=$false)]
    [string]$Format = "plain"  # "plain" o "custom"
)

Write-Host "=== Exportar BD de Produccion ===" -ForegroundColor Cyan
Write-Host ""

# Parsear URL de conexion
function Parse-DatabaseUrl {
    param([string]$Url)
    
    if ($Url -match "postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)") {
        return @{
            User = $matches[1]
            Password = $matches[2]
            Host = $matches[3]
            Port = $matches[4]
            Database = $matches[5]
        }
    } else {
        throw "URL de base de datos invalida: $Url"
    }
}

try {
    # Parsear URL
    Write-Host "[1/3] Parseando URL de conexion..." -ForegroundColor Yellow
    $db = Parse-DatabaseUrl -Url $DatabaseUrl
    
    Write-Host "  Host: $($db.Host):$($db.Port)" -ForegroundColor Gray
    Write-Host "  Database: $($db.Database)" -ForegroundColor Gray
    Write-Host "  Usuario: $($db.User)" -ForegroundColor Gray
    Write-Host ""
    
    # Verificar que pg_dump este disponible
    Write-Host "[2/3] Verificando herramientas de PostgreSQL..." -ForegroundColor Yellow
    $pgDumpPath = Get-Command pg_dump -ErrorAction SilentlyContinue
    
    if (-not $pgDumpPath) {
        Write-Host "ERROR: pg_dump no encontrado. Asegurate de que PostgreSQL este instalado y en el PATH." -ForegroundColor Red
        Write-Host "  Puedes agregarlo manualmente: `$env:Path += ';C:\Program Files\PostgreSQL\15\bin'" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "  pg_dump encontrado: $($pgDumpPath.Source)" -ForegroundColor Green
    Write-Host ""
    
    # Exportar BD
    Write-Host "[3/3] Exportando BD..." -ForegroundColor Yellow
    
    # Configurar variable de entorno para la contraseña (evita prompt interactivo)
    $env:PGPASSWORD = $db.Password
    
    if ($Format -eq "custom") {
        # Formato custom (comprimido, requiere pg_restore para importar)
        $dumpFile = $BackupFile -replace '\.sql$', '.dump'
        Write-Host "  Formato: Custom (comprimido)" -ForegroundColor Gray
        Write-Host "  Archivo de backup: $dumpFile" -ForegroundColor Gray
        
        $dumpArgs = @(
            "-h", $db.Host
            "-p", $db.Port
            "-U", $db.User
            "-d", $db.Database
            "-F", "c"  # Formato custom
            "-f", $dumpFile
            "--no-owner"  # No incluir ownership
            "--no-acl"    # No incluir permisos
            "--verbose"
        )
    } else {
        # Formato SQL plano (texto, más compatible)
        Write-Host "  Formato: SQL plano" -ForegroundColor Gray
        Write-Host "  Archivo de backup: $BackupFile" -ForegroundColor Gray
        
        $dumpArgs = @(
            "-h", $db.Host
            "-p", $db.Port
            "-U", $db.User
            "-d", $db.Database
            "-f", $BackupFile
            "--no-owner"  # No incluir ownership
            "--no-acl"    # No incluir permisos
            "--verbose"
        )
    }
    
    Write-Host "  Ejecutando: pg_dump $($dumpArgs -join ' ')" -ForegroundColor Gray
    & pg_dump $dumpArgs
    
    if ($LASTEXITCODE -ne 0) {
        throw "Error al exportar BD. Codigo de salida: $LASTEXITCODE"
    }
    
    Write-Host ""
    Write-Host "=== Exportacion completada exitosamente ===" -ForegroundColor Green
    
    if ($Format -eq "custom") {
        Write-Host "  Archivo generado: $dumpFile" -ForegroundColor Gray
        Write-Host "  Tamaño: $((Get-Item $dumpFile).Length / 1MB) MB" -ForegroundColor Gray
        Write-Host ""
        Write-Host "  Para importar, usa: .\scripts\importar_bd_local.ps1 -BackupFile `"$dumpFile`"" -ForegroundColor Yellow
    } else {
        Write-Host "  Archivo generado: $BackupFile" -ForegroundColor Gray
        Write-Host "  Tamaño: $((Get-Item $BackupFile).Length / 1MB) MB" -ForegroundColor Gray
        Write-Host ""
        Write-Host "  Para importar, usa: .\scripts\importar_bd_local.ps1 -BackupFile `"$BackupFile`"" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "  Siguiente paso: Copia el archivo a tu maquina local y ejecuta el script de importacion." -ForegroundColor Cyan
    
} catch {
    Write-Host ""
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host $_.ScriptStackTrace -ForegroundColor Red
    exit 1
} finally {
    # Limpiar variables de entorno
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}
