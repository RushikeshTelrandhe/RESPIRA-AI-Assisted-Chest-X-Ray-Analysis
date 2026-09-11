@echo off
Rem ============================================================
Rem RESPIRA - Start backend (FastAPI) + frontend (Vite)
Rem   Backend : backend.app.main:app  (clinical API, /api/v1)
Rem   Frontend: frontend\            (clinical UI, reports + auth)
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

if not exist "%ENV_PY%" (
    echo [ERROR] Python environment not found at %ENV_PY%
    echo         Expected a virtualenv named "env" in the project root.
    exit /b 1
)

if not exist "%FRONT%\package.json" (
    echo [ERROR] Frontend not found at %FRONT%
    echo         Expected the clinical UI in "frontend\" with package.json.
    exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] npm not found on PATH. Install Node.js 18+ first.
    exit /b 1
)

if not exist "%FRONT%\node_modules" (
    echo [WARN] "%FRONT%\node_modules" missing. Running "npm install" first...
    pushd "%FRONT%"
    call npm install
    popd
)

echo Starting RESPIRA backend on %BACKEND_URL% ...
echo Backend logs: %LOG%
start "RESPIRA Backend" cmd /k ""%ENV_PY%" -m uvicorn %BACKEND_MOD% --host 127.0.0.1 --port 8000"

echo Waiting for backend to be ready...
set /a tries=0
:await
set /a tries+=1
if %tries% gtr 60 (
    echo [WARN] Backend did not become ready within 60s. Check the "RESPIRA Backend" window.
    echo        Health: %HEALTH_URL%
    echo        Log   : %LOG%
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

:front
echo Starting RESPIRA frontend on http://localhost:5173 ...
start "RESPIRA Frontend" cmd /k "cd /d "%FRONT%" && npm run dev"

echo.
echo RESPIRA is starting.
echo   Backend  : %BACKEND_URL%  (health %HEALTH_URL%)
echo   Frontend : http://localhost:5173
echo.
echo Keep the two terminal windows open. Close them to stop.
endlocal
