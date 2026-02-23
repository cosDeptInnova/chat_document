@echo off
REM Script para desinstalar el servicio de Celery

set SERVICE_NAME=CosmosNotetakerCelery

echo ========================================
echo Desinstalando servicio Celery
echo ========================================
echo.

REM Detener servicio si está corriendo
echo Deteniendo servicio...
net stop %SERVICE_NAME% >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Servicio detenido
) else (
    echo ℹ️ El servicio no estaba corriendo
)

REM Desinstalar servicio
echo Desinstalando servicio...
where nssm >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    nssm remove %SERVICE_NAME% confirm
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Servicio desinstalado correctamente
    ) else (
        echo ⚠️ Error al desinstalar el servicio
    )
) else (
    echo ⚠️ NSSM no encontrado. Intenta desinstalar manualmente desde servicios de Windows
    echo    o ejecuta: sc delete %SERVICE_NAME%
)

echo.
pause

