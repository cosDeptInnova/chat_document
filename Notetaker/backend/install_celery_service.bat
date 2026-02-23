@echo off
REM Script para instalar Celery como servicio de Windows usando NSSM
REM Requiere NSSM instalado: https://nssm.cc/download

echo ========================================
echo Instalando Celery como servicio de Windows
echo ========================================
echo.

REM Verificar si NSSM está disponible
where nssm >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ NSSM no encontrado en PATH
    echo.
    echo Por favor:
    echo 1. Descarga NSSM desde https://nssm.cc/download
    echo 2. Extrae nssm.exe en una carpeta (ej: C:\nssm)
    echo 3. Añade esa carpeta al PATH o copia nssm.exe aquí
    echo.
    pause
    exit /b 1
)

REM Obtener la ruta del script actual
set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%SCRIPT_DIR%notetaker\Scripts\python.exe
set CELERY_SCRIPT=%SCRIPT_DIR%start_celery.py
set SERVICE_NAME=CosmosNotetakerCelery

echo 📍 Directorio del script: %SCRIPT_DIR%
echo 🐍 Python: %PYTHON_EXE%
echo 📄 Script Celery: %CELERY_SCRIPT%
echo 🔧 Nombre del servicio: %SERVICE_NAME%
echo.

REM Verificar que Python existe
if not exist "%PYTHON_EXE%" (
    echo ❌ No se encontró Python en: %PYTHON_EXE%
    pause
    exit /b 1
)

REM Verificar que el script existe
if not exist "%CELERY_SCRIPT%" (
    echo ❌ No se encontró el script en: %CELERY_SCRIPT%
    pause
    exit /b 1
)

echo ¿Deseas instalar el servicio? (S/N)
set /p INSTALL="> "
if /i not "%INSTALL%"=="S" (
    echo Instalación cancelada.
    pause
    exit /b 0
)

echo.
echo 🔧 Instalando servicio...

REM Detener servicio si ya existe
nssm stop %SERVICE_NAME% >nul 2>&1
nssm remove %SERVICE_NAME% confirm >nul 2>&1

REM Instalar servicio
nssm install %SERVICE_NAME% "%PYTHON_EXE%" "%CELERY_SCRIPT%"

REM Configurar directorio de trabajo
nssm set %SERVICE_NAME% AppDirectory "%SCRIPT_DIR%"

REM Configurar descripción
nssm set %SERVICE_NAME% Description "Cosmos Notetaker - Celery Worker para programar bots de reuniones"

REM Configurar para iniciar automáticamente
nssm set %SERVICE_NAME% Start SERVICE_AUTO_START

REM Configurar modo de inicio (si falla, reiniciar después de 60 segundos)
nssm set %SERVICE_NAME% AppRestartDelay 60000
nssm set %SERVICE_NAME% AppThrottle 1500

REM Configurar variables de entorno si es necesario
REM nssm set %SERVICE_NAME% AppEnvironmentExtra "VAR1=valor1" "VAR2=valor2"

REM Configurar salida de logs
set LOG_DIR=%SCRIPT_DIR%logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
nssm set %SERVICE_NAME% AppStdout "%LOG_DIR%\celery_stdout.log"
nssm set %SERVICE_NAME% AppStderr "%LOG_DIR%\celery_stderr.log"

echo.
echo ✅ Servicio instalado correctamente
echo.
echo Para iniciar el servicio:
echo   net start %SERVICE_NAME%
echo.
echo Para detener el servicio:
echo   net stop %SERVICE_NAME%
echo.
echo Para ver el estado:
echo   sc query %SERVICE_NAME%
echo.
echo Para desinstalar:
echo   Ejecuta uninstall_celery_service.bat
echo.

REM Preguntar si iniciar ahora
echo ¿Deseas iniciar el servicio ahora? (S/N)
set /p START="> "
if /i "%START%"=="S" (
    echo Iniciando servicio...
    net start %SERVICE_NAME%
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Servicio iniciado correctamente
    ) else (
        echo ⚠️ Error al iniciar el servicio. Verifica los logs en %LOG_DIR%
    )
)

echo.
pause

