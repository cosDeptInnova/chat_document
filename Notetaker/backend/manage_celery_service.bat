@echo off
REM Script para gestionar el servicio de Celery

set SERVICE_NAME=CosmosNotetakerCelery

:MENU
cls
echo ========================================
echo Gestión del Servicio Celery
echo ========================================
echo.
echo 1. Iniciar servicio
echo 2. Detener servicio
echo 3. Reiniciar servicio
echo 4. Ver estado
echo 5. Ver logs recientes
echo 6. Salir
echo.
set /p OPTION="Selecciona una opción (1-6): "

if "%OPTION%"=="1" goto START
if "%OPTION%"=="2" goto STOP
if "%OPTION%"=="3" goto RESTART
if "%OPTION%"=="4" goto STATUS
if "%OPTION%"=="5" goto LOGS
if "%OPTION%"=="6" goto END
goto MENU

:START
echo.
echo Iniciando servicio...
net start %SERVICE_NAME%
if %ERRORLEVEL% EQU 0 (
    echo ✅ Servicio iniciado correctamente
) else (
    echo ❌ Error al iniciar el servicio
)
echo.
pause
goto MENU

:STOP
echo.
echo Deteniendo servicio...
net stop %SERVICE_NAME%
if %ERRORLEVEL% EQU 0 (
    echo ✅ Servicio detenido correctamente
) else (
    echo ❌ Error al detener el servicio
)
echo.
pause
goto MENU

:RESTART
echo.
echo Reiniciando servicio...
net stop %SERVICE_NAME%
timeout /t 2 /nobreak >nul
net start %SERVICE_NAME%
if %ERRORLEVEL% EQU 0 (
    echo ✅ Servicio reiniciado correctamente
) else (
    echo ❌ Error al reiniciar el servicio
)
echo.
pause
goto MENU

:STATUS
echo.
echo Estado del servicio:
sc query %SERVICE_NAME%
echo.
pause
goto MENU

:LOGS
echo.
set LOG_DIR=%~dp0logs
if exist "%LOG_DIR%\celery_stdout.log" (
    echo Últimas 20 líneas de stdout:
    echo ========================================
    powershell -Command "Get-Content '%LOG_DIR%\celery_stdout.log' -Tail 20"
    echo.
) else (
    echo ⚠️ No se encontró el archivo de log stdout
)
if exist "%LOG_DIR%\celery_stderr.log" (
    echo Últimas 20 líneas de stderr:
    echo ========================================
    powershell -Command "Get-Content '%LOG_DIR%\celery_stderr.log' -Tail 20"
) else (
    echo ⚠️ No se encontró el archivo de log stderr
)
echo.
pause
goto MENU

:END
exit /b 0

