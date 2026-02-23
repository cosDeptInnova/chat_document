# Script para obtener el native_meeting_id (recall_bot_id) de una reunión
# Uso: .\obtener_native_meeting_id.ps1 -MeetingId "uuid-de-la-reunion"
# O: .\obtener_native_meeting_id.ps1 (para listar todas las reuniones recientes)

param(
    [Parameter(Mandatory=$false)]
    [string]$MeetingId,
    
    [Parameter(Mandatory=$false)]
    [string]$DatabaseUrl = "postgresql://cosmos_user:Cos4321@localhost:5432/cosmos_notetaker",
    
    [Parameter(Mandatory=$false)]
    [int]$Limit = 20
)

function Parse-DatabaseUrl {
    param([string]$Url)
    
    if (-not $Url -or -not $Url.StartsWith("postgresql://")) {
        Write-Host "Error: URL debe comenzar con postgresql://" -ForegroundColor Red
        return $null
    }
    
    $urlPart = $Url.Substring(13)
    $atIndex = $urlPart.IndexOf("@")
    if ($atIndex -eq -1) {
        Write-Host "Error: URL debe tener formato postgresql://user:password@host:port/database" -ForegroundColor Red
        return $null
    }
    
    $credentials = $urlPart.Substring(0, $atIndex)
    $rest = $urlPart.Substring($atIndex + 1)
    
    $colonIndex = $credentials.LastIndexOf(":")
    if ($colonIndex -eq -1) {
        $user = $credentials
        $password = ""
    } else {
        $user = $credentials.Substring(0, $colonIndex)
        $password = $credentials.Substring($colonIndex + 1)
    }
    
    $slashIndex = $rest.IndexOf("/")
    if ($slashIndex -eq -1) {
        $hostPort = $rest
        $database = ""
    } else {
        $hostPort = $rest.Substring(0, $slashIndex)
        $database = $rest.Substring($slashIndex + 1)
    }
    
    $colonIndex = $hostPort.IndexOf(":")
    if ($colonIndex -eq -1) {
        $dbHost = $hostPort
        $port = "5432"
    } else {
        $dbHost = $hostPort.Substring(0, $colonIndex)
        $port = $hostPort.Substring($colonIndex + 1)
    }
    
    return @{
        User = $user
        Password = $password
        Host = $dbHost
        Port = $port
        Database = $database
    }
}

# Parsear URL de base de datos
$dbInfo = Parse-DatabaseUrl -Url $DatabaseUrl
if (-not $dbInfo) {
    exit 1
}

# Verificar si psql está disponible
$psqlPath = Get-Command psql -ErrorAction SilentlyContinue
if (-not $psqlPath) {
    Write-Host "ERROR: psql no está en el PATH" -ForegroundColor Red
    Write-Host "Instala PostgreSQL Client o agrega psql al PATH" -ForegroundColor Yellow
    exit 1
}

Write-Host "=== Obtener Native Meeting ID (recall_bot_id) ===" -ForegroundColor Cyan
Write-Host ""

# Establecer variable de entorno para password
$env:PGPASSWORD = $dbInfo.Password

try {
    if ($MeetingId) {
        # Buscar una reunión específica
        Write-Host "Buscando reunión: $MeetingId" -ForegroundColor Yellow
        Write-Host ""
        
        $query = @"
SELECT 
    id,
    title,
    recall_bot_id as native_meeting_id,
    status,
    scheduled_start_time,
    scheduled_end_time,
    created_at
FROM meetings
WHERE id = '$MeetingId';
"@
        
        $query | & $psqlPath.Path -h $dbInfo.Host -p $dbInfo.Port -U $dbInfo.User -d $dbInfo.Database
        
        Write-Host ""
        Write-Host "Native Meeting ID (recall_bot_id):" -ForegroundColor Cyan
        $nativeIdQuery = @"
SELECT recall_bot_id 
FROM meetings 
WHERE id = '$MeetingId' AND recall_bot_id IS NOT NULL;
"@
        
        $nativeId = $nativeIdQuery | & $psqlPath.Path -h $dbInfo.Host -p $dbInfo.Port -U $dbInfo.User -d $dbInfo.Database -t -A
        if ($nativeId) {
            Write-Host $nativeId.Trim() -ForegroundColor Green
            Write-Host ""
            Write-Host "Usa este ID para consultar VEXA:" -ForegroundColor Yellow
            Write-Host "  ./consultar_vexa_docker.sh $($nativeId.Trim())" -ForegroundColor White
        } else {
            Write-Host "Esta reunión no tiene recall_bot_id asignado aún" -ForegroundColor Yellow
        }
        
    } else {
        # Listar reuniones recientes con sus native_meeting_id
        Write-Host "=== Reuniones recientes con Native Meeting ID ===" -ForegroundColor Cyan
        Write-Host ""
        
        $query = @"
SELECT 
    id as meeting_id,
    title,
    recall_bot_id as native_meeting_id,
    status,
    scheduled_start_time,
    created_at
FROM meetings
WHERE recall_bot_id IS NOT NULL
ORDER BY created_at DESC
LIMIT $Limit;
"@
        
        $query | & $psqlPath.Path -h $dbInfo.Host -p $dbInfo.Port -U $dbInfo.User -d $dbInfo.Database
        
        Write-Host ""
        Write-Host "=== Reuniones SIN Native Meeting ID ===" -ForegroundColor Cyan
        Write-Host ""
        
        $queryNoId = @"
SELECT 
    id as meeting_id,
    title,
    status,
    scheduled_start_time,
    created_at
FROM meetings
WHERE recall_bot_id IS NULL
ORDER BY created_at DESC
LIMIT $Limit;
"@
        
        $queryNoId | & $psqlPath.Path -h $dbInfo.Host -p $dbInfo.Port -U $dbInfo.User -d $dbInfo.Database
        
        Write-Host ""
        Write-Host "Para obtener el native_meeting_id de una reunión específica:" -ForegroundColor Yellow
        Write-Host "  .\obtener_native_meeting_id.ps1 -MeetingId 'uuid-de-la-reunion'" -ForegroundColor White
    }
    
} catch {
    Write-Host "Error ejecutando consulta: $_" -ForegroundColor Red
    exit 1
} finally {
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "=== Completado ===" -ForegroundColor Green
