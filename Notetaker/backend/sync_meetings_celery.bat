@echo off
REM Script para sincronizar reuniones pendientes con Celery

cd /d "%~dp0"

REM Buscar el Python del entorno virtual
set VENV_PYTHON=notetaker\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
    echo No se encontro el entorno virtual
    echo Usando Python del sistema...
    python sync_meetings_celery.py
) else (
    echo Usando Python del entorno virtual
    "%VENV_PYTHON%" sync_meetings_celery.py
)

pause

