@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment Python was not found at .venv\Scripts\python.exe
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Dependency installation failed.
    exit /b %EXIT_CODE%
)

echo.
echo Dependencies installed successfully.
exit /b 0
