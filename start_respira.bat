@echo off
setlocal EnableExtensions EnableDelayedExpansion

Rem ============================================================
Rem RESPIRA - Complete Local Startup
Rem
Rem Backend  : FastAPI
Rem Frontend : Vite + React
Rem Models   : Hugging Face
Rem
Rem Startup sequence:
Rem   1. Validate project structure
Rem   2. Validate Python
Rem   3. Create virtual environment if missing
Rem   4. Install Python dependencies
Rem   5. Validate Node.js / npm
Rem   6. Install frontend dependencies if required
Rem   7. Download missing 512x512 AI models
Rem   8. Start FastAPI backend
Rem   9. Wait for backend health
Rem  10. Start Vite frontend
Rem
Rem Designed for:
Rem   Fresh GitHub clone
Rem   Windows
Rem   No pre-existing Python environment
Rem   No model checkpoints in GitHub
Rem ============================================================


Rem ============================================================
Rem PROJECT PATHS
Rem ============================================================

set "ROOT=%~dp0"
set "FRONT=%ROOT%frontend"
set "BACKEND=%ROOT%backend"
set "SCRIPTS=%ROOT%scripts"

set "VENV=%ROOT%env"
set "ENV_PY=%VENV%\Scripts\python.exe"

set "MODEL_DOWNLOADER=%SCRIPTS%\download_models.py"

set "BACKEND_MOD=backend.app.main:app"
set "BACKEND_URL=http://127.0.0.1:8000"
set "HEALTH_URL=%BACKEND_URL%/api/v1/health"

set "FRONTEND_URL=http://localhost:5173"

set "PYTHONPATH=%ROOT%"
set "PYTHONUTF8=1"

set "REQ_MAIN=%ROOT%requirements.txt"
set "REQ_BACKEND=%ROOT%requirements-backend.txt"
set "REQ_APP=%ROOT%requirements-app.txt"


Rem ============================================================
Rem HEADER
Rem ============================================================

echo.
echo ============================================================
echo                 RESPIRA AI
echo ============================================================
echo.
echo Complete Chest X-Ray Analysis System
echo.
echo Project:
echo   %ROOT%
echo.
echo ============================================================
echo.


Rem ============================================================
Rem CHECK PROJECT STRUCTURE
Rem ============================================================

if not exist "%ROOT%README.md" (
    echo.
    echo [ERROR] README.md not found.
    echo.
    echo This does not appear to be the RESPIRA project root.
    echo.
    pause
    exit /b 1
)

if not exist "%BACKEND%\app\main.py" (
    echo.
    echo [ERROR] FastAPI backend not found.
    echo.
    echo Expected:
    echo   backend\app\main.py
    echo.
    pause
    exit /b 1
)

if not exist "%FRONT%\package.json" (
    echo.
    echo [ERROR] Frontend not found.
    echo.
    echo Expected:
    echo   frontend\package.json
    echo.
    pause
    exit /b 1
)

if not exist "%MODEL_DOWNLOADER%" (
    echo.
    echo [ERROR] Model downloader not found.
    echo.
    echo Expected:
    echo   scripts\download_models.py
    echo.
    pause
    exit /b 1
)


Rem ============================================================
Rem FIND PYTHON
Rem ============================================================

echo [1/7] Checking Python...

where py >nul 2>nul

if not errorlevel 1 (
    set "PY_CMD=py"
    goto :python_found
)

where python >nul 2>nul

if not errorlevel 1 (
    set "PY_CMD=python"
    goto :python_found
)

echo.
echo [ERROR] Python was not found.
echo.
echo Install Python 3.10+ and make sure it is available on PATH.
echo.
echo Recommended:
echo   https://www.python.org/downloads/
echo.
pause
exit /b 1


:python_found

echo Python command:
echo   %PY_CMD%

%PY_CMD% --version

if errorlevel 1 (
    echo.
    echo [ERROR] Python could not be executed.
    echo.
    pause
    exit /b 1
)


Rem ============================================================
Rem CREATE VIRTUAL ENVIRONMENT
Rem ============================================================

echo.
echo [2/7] Checking Python virtual environment...

if exist "%ENV_PY%" (
    echo ✓ Existing virtual environment found.
    goto :venv_ready
)

echo.
echo [INFO] Virtual environment not found.
echo        Creating:
echo        %VENV%
echo.

%PY_CMD% -m venv "%VENV%"

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to create virtual environment.
    echo.
    pause
    exit /b 1
)

if not exist "%ENV_PY%" (
    echo.
    echo [ERROR] Virtual environment creation failed.
    echo.
    pause
    exit /b 1
)

echo ✓ Virtual environment created.


:venv_ready

echo.
echo Python environment:
echo   %ENV_PY%

"%ENV_PY%" --version

if errorlevel 1 (
    echo.
    echo [ERROR] Virtual environment Python is not working.
    echo.
    pause
    exit /b 1
)


Rem ============================================================
Rem INSTALL PYTHON DEPENDENCIES
Rem ============================================================

echo.
echo [3/7] Checking Python dependencies...
echo.

"%ENV_PY%" -m pip --version

if errorlevel 1 (
    echo.
    echo [ERROR] pip is not available.
    echo.
    pause
    exit /b 1
)

echo.
echo Upgrading pip...
"%ENV_PY%" -m pip install --upgrade pip

if errorlevel 1 (
    echo.
    echo [WARN] pip upgrade failed.
    echo        Continuing with existing pip...
)


Rem ------------------------------------------------------------
Rem Main requirements
Rem ------------------------------------------------------------

if exist "%REQ_MAIN%" (
    echo.
    echo Installing main Python requirements...
    "%ENV_PY%" -m pip install -r "%REQ_MAIN%"

    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to install requirements.txt
        echo.
        pause
        exit /b 1
    )
)


Rem ------------------------------------------------------------
Rem Backend requirements
Rem ------------------------------------------------------------

if exist "%REQ_BACKEND%" (
    echo.
    echo Installing backend requirements...
    "%ENV_PY%" -m pip install -r "%REQ_BACKEND%"

    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to install requirements-backend.txt
        echo.
        pause
        exit /b 1
    )
)


Rem ------------------------------------------------------------
Rem Application requirements
Rem ------------------------------------------------------------

if exist "%REQ_APP%" (
    echo.
    echo Installing application requirements...
    "%ENV_PY%" -m pip install -r "%REQ_APP%"

    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to install requirements-app.txt
        echo.
        pause
        exit /b 1
    )
)

echo.
echo ✓ Python dependencies ready.


Rem ============================================================
Rem CHECK NODE.JS / NPM
Rem ============================================================

echo.
echo [4/7] Checking Node.js and npm...
echo.

where node >nul 2>nul

if errorlevel 1 (
    echo.
    echo [ERROR] Node.js was not found.
    echo.
    echo Install Node.js 18+:
    echo   https://nodejs.org/
    echo.
    pause
    exit /b 1
)

where npm >nul 2>nul

if errorlevel 1 (
    echo.
    echo [ERROR] npm was not found.
    echo.
    echo Reinstall Node.js and ensure npm is added to PATH.
    echo.
    pause
    exit /b 1
)

echo Node:
node --version

echo npm:
npm --version


Rem ============================================================
Rem FRONTEND DEPENDENCIES
Rem ============================================================

echo.
echo Checking frontend dependencies...

if not exist "%FRONT%\node_modules" (
    echo.
    echo [INFO] frontend\node_modules not found.
    echo        Running npm install...
    echo.

    pushd "%FRONT%"

    call npm install

    if errorlevel 1 (
        echo.
        echo [ERROR] npm install failed.
        echo.
        popd
        pause
        exit /b 1
    )

    popd

    echo.
    echo ✓ Frontend dependencies installed.
) else (
    echo ✓ Frontend dependencies already installed.
)


Rem ============================================================
Rem DOWNLOAD AI MODELS
Rem ============================================================

echo.
echo [5/7] Checking AI model checkpoints...
echo.

echo Hugging Face model repository:
echo   rushikeshtelrandhe/respira-models
echo.

"%ENV_PY%" "%MODEL_DOWNLOADER%"

if errorlevel 1 (
    echo.
    echo ============================================================
    echo [ERROR] AI model download failed.
    echo ============================================================
    echo.
    echo Required models:
    echo   EfficientNet-B0 512x512
    echo   ViT-B/16 512x512
    echo   Fusion configuration
    echo.
    echo Check:
    echo   1. Internet connection
    echo   2. Hugging Face repository availability
    echo   3. Hugging Face authentication if required
    echo.
    pause
    exit /b 1
)

echo.
echo ✓ AI model checkpoints ready.


Rem ============================================================
Rem CREATE REQUIRED DIRECTORIES
Rem ============================================================

echo.
echo Creating runtime directories...

if not exist "%ROOT%outputs" mkdir "%ROOT%outputs"
if not exist "%ROOT%outputs\efficientnet_b0_512" mkdir "%ROOT%outputs\efficientnet_b0_512"
if not exist "%ROOT%outputs\efficientnet_b0_512\checkpoints" mkdir "%ROOT%outputs\efficientnet_b0_512\checkpoints"
if not exist "%ROOT%outputs\efficientnet_b0_512\evaluation" mkdir "%ROOT%outputs\efficientnet_b0_512\evaluation"

if not exist "%ROOT%outputs\vit_512" mkdir "%ROOT%outputs\vit_512"
if not exist "%ROOT%outputs\vit_512\checkpoints" mkdir "%ROOT%outputs\vit_512\checkpoints"
if not exist "%ROOT%outputs\vit_512\evaluation" mkdir "%ROOT%outputs\vit_512\evaluation"

if not exist "%ROOT%outputs\fusion" mkdir "%ROOT%outputs\fusion"

if not exist "%ROOT%backend_storage" mkdir "%ROOT%backend_storage"

echo ✓ Runtime directories ready.


Rem ============================================================
Rem START BACKEND
Rem ============================================================

echo.
echo [6/7] Starting FastAPI backend...
echo.

echo Backend:
echo   %BACKEND_URL%

start "RESPIRA Backend" cmd /k ^
"cd /d "%ROOT%" && ^
set PYTHONPATH=%ROOT% && ^
set PYTHONUTF8=1 && ^
"%ENV_PY%" -m uvicorn %BACKEND_MOD% --host 127.0.0.1 --port 8000"

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start backend.
    echo.
    pause
    exit /b 1
)


Rem ============================================================
Rem WAIT FOR BACKEND
Rem ============================================================

echo.
echo Waiting for backend health endpoint...

set /a tries=0

:await_backend

set /a tries+=1

if !tries! GTR 60 (
    echo.
    echo [WARN] Backend did not become ready within 60 seconds.
    echo.
    echo Check the "RESPIRA Backend" terminal.
    echo.
    echo Health endpoint:
    echo   %HEALTH_URL%
    echo.
    goto :start_frontend
)

powershell -NoProfile -Command ^
"try { $r=Invoke-RestMethod -Uri '%HEALTH_URL%' -TimeoutSec 2; if ($r.status -eq 'ok') { 'ok' } else { 'ready' } } catch { 'no' }" ^
> "%TEMP%\respira_ready.txt"

set "READY="

set /p READY=<"%TEMP%\respira_ready.txt"

if /I "!READY!"=="ok" (
    echo ✓ Backend is ready.
    goto :start_frontend
)

if /I "!READY!"=="ready" (
    echo ✓ Backend responded.
    goto :start_frontend
)

timeout /t 1 /nobreak >nul

goto :await_backend


Rem ============================================================
Rem START FRONTEND
Rem ============================================================

:start_frontend

echo.
echo [7/7] Starting frontend...
echo.

echo Frontend:
echo   %FRONTEND_URL%

start "RESPIRA Frontend" cmd /k ^
"cd /d "%FRONT%" && npm run dev"

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start frontend.
    echo.
    pause
    exit /b 1
)


Rem ============================================================
Rem FINAL MESSAGE
Rem ============================================================

echo.
echo ============================================================
echo                  RESPIRA STARTED
echo ============================================================
echo.
echo Backend:
echo   %BACKEND_URL%
echo.
echo Health:
echo   %HEALTH_URL%
echo.
echo Frontend:
echo   %FRONTEND_URL%
echo.
echo AI Models:
echo   Hugging Face
echo   rushikeshtelrandhe/respira-models
echo.
echo Models:
echo   EfficientNet-B0 512x512
echo   ViT-B/16 512x512
echo   Probability Fusion
echo.
echo ============================================================
echo.
echo Open in browser:
echo   http://localhost:5173
echo.
echo Keep the Backend and Frontend windows open.
echo Close them to stop RESPIRA.
echo.

endlocal
exit /b 0