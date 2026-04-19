@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Creating .venv...
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv .venv
    ) else (
        where python >nul 2>nul
        if not errorlevel 1 (
            python -m venv .venv
        ) else (
            echo.
            echo Python 3 was not found on PATH.
            echo Install Python 3, or use the Windows py launcher, then rerun this script.
            exit /b 1
        )
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Virtual environment creation failed.
    exit /b 1
)

echo.
echo Installing dependencies into .venv...
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
