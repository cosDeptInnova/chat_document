@echo off
REM Script para activar el entorno virtual notetaker
cd /d "%~dp0"
call notetaker\Scripts\activate.bat
echo.
echo ========================================
echo Entorno virtual 'notetaker' activado
echo ========================================
echo.
echo Para desactivar, escribe: deactivate
echo.

