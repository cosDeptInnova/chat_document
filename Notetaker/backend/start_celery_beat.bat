@echo off
REM Script para iniciar Celery Beat en Windows (tareas periodicas: check-stuck-meetings, check-and-finalize-completed-meetings, check-system-health)
cd /d "%~dp0"

REM Usar Python del venv si existe; si no, usar el del PATH (ej. conda activate cosmos)
set "VENV_PYTHON=%~dp0notetaker\Scripts\python.exe"
if exist "%VENV_PYTHON%" (
    set "PYTHON_CMD=%VENV_PYTHON%"
    echo Usando entorno virtual: notetaker
) else (
    set "PYTHON_CMD=python"
    echo Usando Python del PATH (ej. conda cosmos)
)

echo Iniciando Celery Beat...
"%PYTHON_CMD%" -m celery -A app.celery_app beat --loglevel=info

if errorlevel 1 (
    echo.
    echo Error al iniciar Celery Beat. Comprueba: Redis en marcha, dependencias instaladas en el env activo.
    pause
)
