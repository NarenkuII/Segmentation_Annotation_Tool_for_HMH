@echo off
setlocal

cd /d "%~dp0"

echo Demarrage de l'outil HMH de segmentation LSF...
echo Workspace: %CD%
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo Python est introuvable dans le PATH.
  echo Installe Python 3 et verifie que "python" fonctionne dans un terminal.
  pause
  exit /b 1
)

start "" http://127.0.0.1:8000/index.html
python app.py

echo.
echo Serveur arrete.
pause
