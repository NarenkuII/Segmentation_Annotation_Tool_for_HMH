@echo off
setlocal

cd /d "%~dp0"

echo Starting LSF annotation tool...
echo Workspace: %CD%
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found in PATH.
  echo Install Python 3 and make sure "python" works in a terminal.
  pause
  exit /b 1
)

start "" http://127.0.0.1:8000/index.html
python app.py

echo.
echo Server stopped.
pause
