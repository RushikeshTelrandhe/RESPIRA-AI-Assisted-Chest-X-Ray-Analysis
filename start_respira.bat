@echo off
Rem ============================================================
Rem RESPIRA - Start backend (FastAPI) + frontend (Vite)
Rem   Backend : backend.app.main:app
Rem   Frontend: frontend\
Rem
Rem Startup sequence:
Rem   1. Validate Python / frontend / npm
Rem   2. Download missing AI model checkpoints
Rem   3. Start FastAPI backend
Rem   4. Wait for backend health
Rem   5. Start Vite frontend
Rem ============================================================

setlocal

set "ROOT=%~dp0"
set "ENV_PY=%ROOT%env\Scripts\python.exe"
set "FRONT=%ROOT%frontend"
set "PYTHONUTF8=1"
set "PYTHONPATH=%ROOT%"

set "BACKEND_MOD=backend.app.main:app"
set "BACKEND_URL=http://127.0.0.1:8000"
set "HEALTH_URL=%BACKEND_URL%/api/v1/health"
set "LOG=%ROOT%backend.log"
set "MODEL_DOWNLOADER=%ROOT%scripts\download_models.py"


Rem ============================================================
Rem Check Python environment
Rem ============================================================

if not exist "%ENV_PY%" (
    echo.
    echo [ERROR] Python environment not found:
    echo         %ENV_PY%
    echo.
    echo Expected a virtualenv named "env" in the project root.
    exit /b 1
)


Rem ============================================================
Rem Check frontend
Rem ============================================================

if not exist "%FRONT%\package.json" (
    echo.
    echo [ERROR] Frontend not found:
    echo         %FRONT%
    echo.
    echo Expected the clinical UI in "frontend\" with package.json.
    exit /b 1
)


Rem ============================================================
Rem Check npm
Rem ============================================================

where npm >nul 2>nul

if errorlevel 1 (
    echo.
    echo [ERROR] npm not found on PATH.
    echo         Install Node.js 18+ first.
    exit /b 1
)


Rem ============================================================
Rem Check model downloader
Rem ============================================================

if not exist "%MODEL_DOWNLOADER%" (
    echo.
    echo [ERROR] Model downloader not found:
    echo         %MODEL_DOWNLOADER%
    echo.
    echo Expected scripts\download_models.py
    exit /b 1
)


Rem ============================================================
Rem Install frontend dependencies if necessary
Rem ============================================================

if not exist "%FRONT%\node_modules" (
    echo.
    echo [WARN] Frontend node_modules missing.
    echo        Running npm install...
    echo.

    pushd "%FRONT%"
    call npm install

    if errorlevel 1 (
        echo.
        echo [ERROR] npm install failed.
        popd
        exit /b 1
    )

    popd
)


Rem ============================================================
Rem Check / Download AI models
Rem ============================================================

echo.
echo ============================================================
echo RESPIRA - Checking AI model checkpoints
echo ============================================================
echo.

"%ENV_PY%" "%MODEL_DOWNLOADER%"

if errorlevel 1 (
    echo.
    echo ============================================================
    echo [ERROR] Required AI models are not available.
    echo ============================================================
    echo.
    echo Respira cannot start without the required model checkpoints.
    echo.
    pause
    exit /b 1
)

echo.
echo Model checkpoints ready.
echo.


Rem ============================================================
Rem Start backend
Rem ============================================================

echo Starting RESPIRA backend on %BACKEND_URL% ...
echo Backend logs: %LOG%

start "RESPIRA Backend" cmd /k ""%ENV_PY%" -m uvicorn %BACKEND_MOD% --host 127.0.0.1 --port 8000"


Rem ============================================================
Rem Wait for backend
Rem ============================================================

echo.
echo Waiting for backend to be ready...

set /a tries=0

:await

set /a tries+=1

if %tries% gtr 60 (
    echo.
    echo [WARN] Backend did not become ready within 60 seconds.
    echo.
    echo Check the "RESPIRA Backend" window.
    echo Health: %HEALTH_URL%
    echo Log   : %LOG%
    goto :front
)

powershell -NoProfile -Command "try { (Invoke-RestMethod -Uri '%HEALTH_URL%' -TimeoutSec 2).status } catch { 'no' }" > "%TEMP%\respira_ready.txt"

set /p ready=<"%TEMP%\respira_ready.txt"

if "%ready%"=="ok" (
    echo Backend ready.
    goto :front
)

timeout /t 1 /nobreak >nul
goto :await


Rem ============================================================
Rem Start frontend
Rem ============================================================

:front

echo.
echo Starting RESPIRA frontend on http://localhost:5173 ...

start "RESPIRA Frontend" cmd /k "cd /d "%FRONT%" && npm run dev"


Rem ============================================================
Rem Done
Rem ============================================================

echo.
echo ============================================================
echo RESPIRA is starting.
echo ============================================================
echo.
echo   Backend  : %BACKEND_URL%
echo   Health   : %HEALTH_URL%
echo   Frontend : http://localhost:5173
echo.
echo Keep the two terminal windows open.
echo Close them to stop RESPIRA.
echo.

endlocal