@echo off
REM Script para iniciar Celery worker en Windows
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

echo Iniciando Celery worker...
"%PYTHON_CMD%" -m celery -A app.celery_app worker --loglevel=info --pool=solo -Q celery,summary_queue -c 1 --prefetch-multiplier=1

if errorlevel 1 (
    echo.
    echo Error al iniciar Celery. Comprueba: Redis en marcha, dependencias instaladas en el env activo.
    pause
)
