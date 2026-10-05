@echo off
setlocal EnableExtensions EnableDelayedExpansion
title RESPIRA - AI Assisted Chest X-Ray Analysis

REM ============================================================
REM  RESPIRA one-click starter (double-click from Explorer).
REM  - Project root is ALWAYS the folder containing this BAT.
REM  - Never relies on the caller's working directory.
REM  - Uses env\Scripts\python.exe directly (no activation).
REM  - Every major step checks errorlevel and stops with a
REM    clear message instead of silently terminating.
REM ============================================================

cd /d "%~dp0"
if errorlevel 1 (
    echo [ERROR] Could not change to the BAT directory: "%~dp0"
    echo Start this file by double-clicking it in Windows Explorer.
    pause
    exit /b 1
)

set "ROOT=%CD%"
set "ENV_DIR=%ROOT%\env"
set "PYTHON=%ENV_DIR%\Scripts\python.exe"
set "FRONTEND=%ROOT%\frontend"
set "DOWNLOADER=%ROOT%\scripts\download_models.py"

set "BACKEND_HOST=127.0.0.1"
set "BACKEND_PORT=8000"
set "FRONTEND_PORT=5173"

set "BACKEND_URL=http://%BACKEND_HOST%:%BACKEND_PORT%"
set "HEALTH_URL=%BACKEND_URL%/api/v1/health"
set "FRONTEND_URL=http://localhost:%FRONTEND_PORT%"

REM Required by "backend.app.main:app" top-level package import.
set "PYTHONUTF8=1"
set "PYTHONPATH=%ROOT%"

echo.
echo ============================================================
echo      RESPIRA - AI ASSISTED CHEST X-RAY ANALYSIS
echo ============================================================
echo.
echo Project:
echo   %ROOT%
echo.
echo Backend:
echo   %BACKEND_URL%
echo.
echo Frontend:
echo   %FRONTEND_URL%
echo.
echo ============================================================
echo.

REM ============================================================
REM  [1/8] CHECK PYTHON
REM ============================================================
echo [1/8] Checking Python...
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo Component : Python check ^(where python^)
    echo Fix       : Install Python 3.11 - 3.13 from python.org,
    echo             tick "Add python.exe to PATH", then double-click
    echo             this BAT again.
    echo.
    pause
    exit /b 1
)

python --version
if errorlevel 1 (
    echo [ERROR] "python --version" failed.
    echo Component : Python installation
    echo Fix       : Reinstall Python 3.11 - 3.13 with PATH enabled.
    echo.
    pause
    exit /b 1
)

echo.
echo Python found.
echo.

REM ============================================================
REM  [2/8] CREATE / CHECK VIRTUAL ENVIRONMENT
REM ============================================================
echo [2/8] Creating/checking virtual environment...
echo.

if not exist "%PYTHON%" (
    echo Virtual environment not found. Creating:
    echo   %ENV_DIR%
    echo.
    python -m venv "%ENV_DIR%"
    if errorlevel 1 (
        echo.
        echo [ERROR] Could not create the virtual environment.
        echo Component : python -m venv "%ENV_DIR%"
        echo Fix       : Make sure Python 3.11 - 3.13 is installed with
        echo             pip, then delete the "env" folder and retry.
        echo.
        pause
        exit /b 1
    )
    echo.
    echo Virtual environment created.
    echo.
)

if not exist "%PYTHON%" (
    echo.
    echo [ERROR] Virtual environment Python is still missing:
    echo   %PYTHON%
    echo Component : virtual environment layout
    echo Fix       : Delete the "env" folder and run this BAT again.
    echo.
    pause
    exit /b 1
)

"%PYTHON%" --version
if errorlevel 1 (
    echo.
    echo [ERROR] The virtual environment Python does not run:
    echo   %PYTHON%
    echo Component : env\Scripts\python.exe
    echo Fix       : Delete the "env" folder and run this BAT again.
    echo.
    pause
    exit /b 1
)

echo.
echo Python environment ready.
echo.

REM ============================================================
REM  [3/8] INSTALL PYTHON DEPENDENCIES (idempotent)
REM  Uses requirements-backend.txt + requirements-app.txt.
REM  requirements.txt is intentionally NOT used: it is the
REM  research/training bundle and contains a CUDA index-url line
REM  that breaks plain "pip install -r" on a fresh machine.
REM ============================================================
echo [3/8] Installing Python dependencies...
echo.

if not exist "%ROOT%\requirements-backend.txt" (
    echo [ERROR] requirements-backend.txt not found in:
    echo   %ROOT%
    echo Component : Python dependency files
    echo Fix       : Clone the repository again - files are missing.
    echo.
    pause
    exit /b 1
)

if not exist "%ROOT%\requirements-app.txt" (
    echo [ERROR] requirements-app.txt not found in:
    echo   %ROOT%
    echo Component : Python dependency files
    echo Fix       : Clone the repository again - files are missing.
    echo.
    pause
    exit /b 1
)

REM Fast path: everything already importable -> skip pip entirely.
set "DEPS_OK=0"
"%PYTHON%" -c "import fastapi, uvicorn, torch, huggingface_hub" >nul 2>&1
if not errorlevel 1 (
    set "DEPS_OK=1"
)

if "%DEPS_OK%"=="1" (
    echo Core Python packages already present - skipping pip install.
    echo.
) else (
    echo Upgrading pip...
    "%PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 (
        echo [WARNING] pip upgrade failed - continuing anyway.
        echo Component : pip --upgrade
        echo.
    )

    echo.
    echo Installing backend requirements ^(requirements-backend.txt^)...
    echo.
    "%PYTHON%" -m pip install -r "%ROOT%\requirements-backend.txt"
    if errorlevel 1 (
        echo.
        echo [ERROR] Backend requirements installation failed.
        echo Component : pip install -r requirements-backend.txt
        echo Fix       : Check your internet connection, then run:
        echo             "%PYTHON%" -m pip install -r requirements-backend.txt
        echo           and read the red error lines.
        echo.
        pause
        exit /b 1
    )

    echo.
    echo Installing app/ML requirements ^(requirements-app.txt^)...
    echo.
    "%PYTHON%" -m pip install -r "%ROOT%\requirements-app.txt"
    if errorlevel 1 (
        echo.
        echo [ERROR] App requirements installation failed.
        echo Component : pip install -r requirements-app.txt
        echo Fix       : Check your internet connection, then run:
        echo             "%PYTHON%" -m pip install -r requirements-app.txt
        echo           and read the red error lines.
        echo.
        pause
        exit /b 1
    )

    echo.
    echo Verifying core imports...
    echo.
    "%PYTHON%" -c "import fastapi, uvicorn, torch, huggingface_hub"
    if errorlevel 1 (
        echo.
        echo [ERROR] Core Python packages still not importable.
        echo Component : fastapi / uvicorn / torch / huggingface_hub
        echo Fix       : Run the two pip install commands above manually
        echo             and read the error output.
        echo.
        pause
        exit /b 1
    )
)

echo.
echo Python dependencies ready.
echo.

REM ============================================================
REM  [4/8] CHECK NODE.JS AND FRONTEND
REM  NOTE: npm MUST be invoked with CALL because npm is a batch
REM  file (npm.cmd). Without CALL, control never returns to this
REM  script - this was the bug that killed the old BAT.
REM ============================================================
echo [4/8] Checking Node.js and frontend...
echo.

where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js was not found.
    echo Component : node ^(where node^)
    echo Fix       : Install Node.js 18+ LTS from nodejs.org,
    echo             then double-click this BAT again.
    echo.
    pause
    exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm was not found.
    echo Component : npm ^(where npm^)
    echo Fix       : Reinstall Node.js 18+ LTS ^(npm ships with it^).
    echo.
    pause
    exit /b 1
)

echo Node:
call node --version
if errorlevel 1 (
    echo [ERROR] "node --version" failed.
    echo Component : Node.js installation
    echo Fix       : Reinstall Node.js 18+ LTS.
    echo.
    pause
    exit /b 1
)

echo npm:
call npm --version
if errorlevel 1 (
    echo [ERROR] "npm --version" failed.
    echo Component : npm installation
    echo Fix       : Reinstall Node.js 18+ LTS.
    echo.
    pause
    exit /b 1
)
echo.

if not exist "%FRONTEND%\package.json" (
    echo [ERROR] frontend\package.json not found:
    echo   %FRONTEND%\package.json
    echo Component : frontend checkout
    echo Fix       : Clone the repository again - files are missing.
    echo.
    pause
    exit /b 1
)

if not exist "%FRONTEND%\node_modules" (
    echo Installing frontend dependencies - this may take a few minutes.
    echo.
    pushd "%FRONTEND%"
    if errorlevel 1 (
        echo [ERROR] Could not enter the frontend directory:
        echo   %FRONTEND%
        echo.
        pause
        exit /b 1
    )
    call npm install
    if errorlevel 1 (
        echo.
        echo [ERROR] npm install failed.
        echo Component : npm install ^(inside frontend\^)
        echo Fix       : Check your internet connection, delete
        echo             frontend\node_modules, and run this BAT again.
        echo.
        popd
        pause
        exit /b 1
    )
    popd
    echo.
    echo Frontend dependencies installed.
    echo.
) else (
    echo Frontend dependencies already present.
    echo.
)

echo Frontend ready.
echo.

REM ============================================================
REM  [5/8] DOWNLOAD / CHECK AI MODELS (Hugging Face)
REM  Skips the downloader when all 5 expected files are present.
REM ============================================================
echo [5/8] Downloading/checking AI models...
echo.

if not exist "%DOWNLOADER%" (
    echo [ERROR] Model downloader not found:
    echo   %DOWNLOADER%
    echo Component : scripts\download_models.py
    echo Fix       : Clone the repository again - files are missing.
    echo.
    pause
    exit /b 1
)

set "M1=%ROOT%\outputs\efficientnet_b0_512\checkpoints\best_model.pth"
set "M2=%ROOT%\outputs\vit_512\checkpoints\best_model.pth"
set "M3=%ROOT%\outputs\efficientnet_b0_512\evaluation\metrics.json"
set "M4=%ROOT%\outputs\vit_512\evaluation\metrics.json"
set "M5=%ROOT%\outputs\fusion\fusion_config.json"

set "MODELS_OK=1"
if not exist "%M1%" set "MODELS_OK=0"
if not exist "%M2%" set "MODELS_OK=0"
if not exist "%M3%" set "MODELS_OK=0"
if not exist "%M4%" set "MODELS_OK=0"
if not exist "%M5%" set "MODELS_OK=0"

if "%MODELS_OK%"=="1" (
    echo All 5 model files already present - skipping download.
    echo.
) else (
    echo Hugging Face repository:
    echo   rushikeshtelrandhe/respira-models
    echo.
    echo Downloading missing model files...
    echo.
    "%PYTHON%" "%DOWNLOADER%"
    if errorlevel 1 (
        echo.
        echo ============================================================
        echo [ERROR] MODEL DOWNLOAD FAILED
        echo ============================================================
        echo Component : env\Scripts\python.exe scripts\download_models.py
        echo.
        echo Check your internet connection and Hugging Face access, then
        echo run this BAT again. The backend will NOT be started with
        echo missing models.
        echo.
        echo Required files:
        echo   outputs\efficientnet_b0_512\checkpoints\best_model.pth
        echo   outputs\vit_512\checkpoints\best_model.pth
        echo   outputs\efficientnet_b0_512\evaluation\metrics.json
        echo   outputs\vit_512\evaluation\metrics.json
        echo   outputs\fusion\fusion_config.json
        echo.
        pause
        exit /b 1
    )
    echo.
)

echo AI models ready.
echo.

REM ============================================================
REM  Local .env (fresh-clone convenience; backend has safe
REM  defaults if this step is skipped).
REM ============================================================
if not exist "%ROOT%\.env" (
    if exist "%ROOT%\.env.example" (
        echo No .env found - creating one from .env.example.
        copy /y "%ROOT%\.env.example" "%ROOT%\.env" >nul
        if errorlevel 1 (
            echo [WARNING] Could not create .env - continuing with defaults.
            echo.
        ) else (
            echo Created .env from .env.example.
            echo.
        )
    )
)

REM ============================================================
REM  [6/8] START BACKEND (separate terminal window)
REM  A tiny launcher file is generated so the START command needs
REM  no fragile nested quoting at all.
REM ============================================================
echo [6/8] Starting FastAPI backend...
echo.
echo Backend:
echo   %BACKEND_URL%
echo.

set "BACKEND_CMD=%TEMP%\respira_backend.cmd"
>  "%BACKEND_CMD%" echo @echo off
>> "%BACKEND_CMD%" echo title RESPIRA BACKEND
>> "%BACKEND_CMD%" echo cd /d "%ROOT%"
>> "%BACKEND_CMD%" echo set "PYTHONPATH=%ROOT%"
>> "%BACKEND_CMD%" echo set "PYTHONUTF8=1"
>> "%BACKEND_CMD%" echo "%PYTHON%" -m uvicorn backend.app.main:app --host %BACKEND_HOST% --port %BACKEND_PORT%
>> "%BACKEND_CMD%" echo pause

if not exist "%BACKEND_CMD%" (
    echo [ERROR] Could not write the backend launcher:
    echo   %BACKEND_CMD%
    echo Component : backend launcher generation
    echo Fix       : Make sure %%TEMP%% is writable.
    echo.
    pause
    exit /b 1
)

start "RESPIRA BACKEND" /D "%ROOT%" cmd /k "%BACKEND_CMD%"
if errorlevel 1 (
    echo [ERROR] Could not open the backend terminal window.
    echo Component : start RESPIRA BACKEND
    echo Fix       : Try starting the backend manually in a new terminal:
    echo             "%PYTHON%" -m uvicorn backend.app.main:app --host %BACKEND_HOST% --port %BACKEND_PORT%
    echo.
    pause
    exit /b 1
)

echo Backend terminal opened. Waiting for health check...
echo   %HEALTH_URL%
echo.

set "BACKEND_READY=0"
for /L %%N in (1,1,60) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r=Invoke-RestMethod -Uri '%HEALTH_URL%' -TimeoutSec 2; if ($r.status -eq 'ok') { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
    if not errorlevel 1 (
        set "BACKEND_READY=1"
        goto BACKEND_READY
    )
    echo Waiting for backend... %%N/60
    timeout /t 1 /nobreak >nul
)

:BACKEND_READY
if "%BACKEND_READY%"=="0" (
    echo.
    echo ============================================================
    echo [ERROR] BACKEND DID NOT BECOME HEALTHY IN 60 SECONDS
    echo ============================================================
    echo Component : %HEALTH_URL%
    echo.
    echo The backend terminal window is still open - read the red error
    echo lines there. Common causes: missing Python packages, port 8000
    echo already in use, or missing model files.
    echo.
    echo Frontend startup is UNSAFE without a healthy backend, so this
    echo script is STOPPING here. Fix the backend error and run this
    echo BAT again.
    echo.
    pause
    exit /b 1
)

echo.
echo Backend is READY.
echo.

REM ============================================================
REM  [7/8] START FRONTEND (separate terminal window)
REM ============================================================
echo [7/8] Starting Vite frontend...
echo.

set "FRONTEND_CMD=%TEMP%\respira_frontend.cmd"
>  "%FRONTEND_CMD%" echo @echo off
>> "%FRONTEND_CMD%" echo title RESPIRA FRONTEND
>> "%FRONTEND_CMD%" echo cd /d "%FRONTEND%"
>> "%FRONTEND_CMD%" echo call npm run dev -- --host %BACKEND_HOST% --port %FRONTEND_PORT%
>> "%FRONTEND_CMD%" echo pause

if not exist "%FRONTEND_CMD%" (
    echo [ERROR] Could not write the frontend launcher:
    echo   %FRONTEND_CMD%
    echo Component : frontend launcher generation
    echo Fix       : Make sure %%TEMP%% is writable.
    echo.
    pause
    exit /b 1
)

start "RESPIRA FRONTEND" /D "%FRONTEND%" cmd /k "%FRONTEND_CMD%"
if errorlevel 1 (
    echo [ERROR] Could not open the frontend terminal window.
    echo Component : start RESPIRA FRONTEND
    echo Fix       : Open a new terminal in frontend\ and run:
    echo             npm run dev -- --host %BACKEND_HOST% --port %FRONTEND_PORT%
    echo.
    pause
    exit /b 1
)

echo Frontend terminal opened. Waiting for port %FRONTEND_PORT%...
echo.

set "FRONTEND_READY=0"
for /L %%N in (1,1,30) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $c=New-Object Net.Sockets.TcpClient; $c.Connect('%BACKEND_HOST%', %FRONTEND_PORT%); $c.Close(); exit 0 } catch { exit 1 }" >nul 2>&1
    if not errorlevel 1 (
        set "FRONTEND_READY=1"
        goto FRONTEND_READY
    )
    echo Waiting for frontend... %%N/30
    timeout /t 1 /nobreak >nul
)

:FRONTEND_READY
if "%FRONTEND_READY%"=="0" (
    echo [WARNING] Frontend port did not open within 30 seconds.
    echo The frontend terminal is still open - check it for errors.
    echo Continuing to open the browser anyway.
    echo.
) else (
    echo Frontend is responding.
    echo.
)

REM ============================================================
REM  [8/8] OPEN BROWSER (only after frontend was started)
REM ============================================================
echo [8/8] Opening RESPIRA in the browser...
echo.

start "" "%FRONTEND_URL%"
if errorlevel 1 (
    echo [WARNING] Could not open the browser automatically.
    echo Please open this URL manually:
    echo   %FRONTEND_URL%
    echo.
) else (
    echo Browser opened:
    echo   %FRONTEND_URL%
    echo.
)

echo.
echo ============================================================
echo              RESPIRA IS RUNNING
echo ============================================================
echo.
echo Backend:
echo   %BACKEND_URL%
echo.
echo API Docs:
echo   %BACKEND_URL%/docs
echo.
echo Health:
echo   %HEALTH_URL%
echo.
echo Frontend:
echo   %FRONTEND_URL%
echo.
echo Keep the Backend and Frontend terminal windows open.
echo Closing them stops RESPIRA.
echo.
echo ============================================================
echo.

pause
endlocal
