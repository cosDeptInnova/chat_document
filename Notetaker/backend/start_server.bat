@echo off
REM Script para iniciar el servidor FastAPI
cd /d "%~dp0"

echo ========================================
echo Iniciando servidor FastAPI Notetaker
echo ========================================
echo.

REM Usar Python del venv si existe; si no, usar el del PATH (ej. conda activate cosmos)
set "VENV_PYTHON=%~dp0notetaker\Scripts\python.exe"
if exist "%VENV_PYTHON%" (
    set "PYTHON_CMD=%VENV_PYTHON%"
    echo Usando entorno virtual: notetaker
) else (
    set "PYTHON_CMD=python"
    echo Usando Python del PATH (ej. conda cosmos)
)

echo Iniciando servidor en http://0.0.0.0:7000
echo Presiona Ctrl+C para detener el servidor
echo.
"%PYTHON_CMD%" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 7000

if errorlevel 1 (
    echo.
    echo Error al iniciar el servidor. Comprueba dependencias instaladas en el env activo.
    pause
)
