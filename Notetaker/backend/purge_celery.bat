@echo off
REM Script para purgar todas las tareas de Celery

cd /d "%~dp0"

REM Buscar el Python del entorno virtual
set VENV_PYTHON=notetaker\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
    echo ⚠️ No se encontro el entorno virtual
    echo Usando Python del sistema...
    python purge_celery.py
) else (
    echo ✅ Usando Python del entorno virtual
    "%VENV_PYTHON%" purge_celery.py
)

pause

