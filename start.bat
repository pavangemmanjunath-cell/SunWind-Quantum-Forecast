@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%~dp0"
set "VENV=%LOCALAPPDATA%\SunWind-Quantum-Forecast\venv"
if not defined LOCALAPPDATA set "VENV=%ROOT%backend\.venv"
title SunWind-Quantum-Forecast setup

echo.
echo ============================================================
echo  SunWind-Quantum-Forecast - local research dashboard
echo ============================================================
echo  Python API : http://127.0.0.1:8000
echo  Dashboard  : http://127.0.0.1:5173
echo  API docs   : http://127.0.0.1:8000/docs
echo.
echo  Dataset    : region-parameterized synthetic series, generated
echo               by backend\app\data\synthetic.py
echo  Results    : committed experiment outputs in results\
echo  Note       : no live grid data or quantum hardware is used.
echo.

where node >nul 2>&1
if errorlevel 1 goto :missing_node
for /f %%V in ('node -p "process.versions.node.split('.')[0]"') do set "NODE_MAJOR=%%V"
if %NODE_MAJOR% LSS 20 goto :old_node

set "PYTHON="
where py >nul 2>&1
if not errorlevel 1 (
  py -3 --version >nul 2>&1
  if not errorlevel 1 set "PYTHON=py -3"
)
if not defined PYTHON (
  where python >nul 2>&1
  if not errorlevel 1 set "PYTHON=python"
)
if not defined PYTHON goto :missing_python
%PYTHON% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 goto :old_python

if not exist "%VENV%\Scripts\python.exe" (
  echo [1/5] Creating backend virtual environment...
  %PYTHON% -m venv "%VENV%"
  if errorlevel 1 (
    echo Standard venv bootstrap failed; creating an isolated environment and bootstrapping pip...
    %PYTHON% -m venv --clear --without-pip "%VENV%"
    if errorlevel 1 goto :venv_failed
    %PYTHON% -m pip --python "%VENV%" install --disable-pip-version-check pip
    if errorlevel 1 goto :venv_failed
  )
)

echo [2/5] Checking and installing backend dependencies...
"%VENV%\Scripts\python.exe" -m pip install --disable-pip-version-check -r "backend\requirements-dev.txt"
if errorlevel 1 goto :backend_install_failed

if not exist "frontend\node_modules\vite\bin\vite.js" (
  echo [3/5] Installing frontend dependencies from package-lock.json...
  pushd frontend
  call npm.cmd ci --no-audit --no-fund
  if errorlevel 1 (
    popd
    goto :frontend_install_failed
  )
  popd
) else (
  echo [3/5] Frontend dependencies already installed.
)

powershell -NoProfile -Command "foreach ($port in @(8000,5173)) { if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { Write-Error ('Port ' + $port + ' is already in use. Close that service, then retry.'); exit 1 } }"
if errorlevel 1 goto :port_in_use

echo [4/5] Starting API and dashboard...
start "SunWind API" /D "%ROOT%backend" cmd /k ""%VENV%\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
start "SunWind dashboard" /D "%ROOT%frontend" cmd /k "npm.cmd run dev -- --host 127.0.0.1"

echo [5/5] Waiting for both services to become ready...
powershell -NoProfile -Command "$deadline=(Get-Date).AddSeconds(120); do { $api=$false; $web=$false; try { Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/health' -TimeoutSec 2 | Out-Null; $api=$true } catch {}; try { Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:5173/' -TimeoutSec 2 | Out-Null; $web=$true } catch {}; if ($api -and $web) { exit 0 }; Start-Sleep -Seconds 2 } while ((Get-Date) -lt $deadline); Write-Error 'Startup timed out. Inspect the API and dashboard terminal windows for errors.'; exit 1"
if errorlevel 1 goto :startup_failed

echo.
echo Services are ready. Opening the dashboard...
start "" "http://127.0.0.1:5173"
echo.
echo The browser shows the stored research replay and live local API status.
echo Experiment regeneration is intentionally separate; see README.md.
echo Close the two service windows to stop the API and dashboard.
echo.
pause
exit /b 0

:missing_node
echo ERROR: Node.js 20 or newer is required. Install it from https://nodejs.org/ and retry.
goto :failed
:old_node
echo ERROR: Node.js 20 or newer is required. Detected major version %NODE_MAJOR%.
goto :failed
:missing_python
echo ERROR: Python 3.10 or newer is required. Install Python and retry.
goto :failed
:old_python
echo ERROR: Python 3.10 or newer is required. The selected Python is too old.
goto :failed
:venv_failed
echo ERROR: Could not create the isolated Python environment at "%VENV%". Confirm Python and the user profile are writable.
goto :failed
:backend_install_failed
echo ERROR: Backend dependency installation failed. Check the Python and pip output above.
goto :failed
:frontend_install_failed
echo ERROR: Frontend dependency installation failed. Check the Node.js and npm output above.
goto :failed
:port_in_use
echo ERROR: Port 8000 or 5173 is already in use. Stop that service and retry.
goto :failed
:startup_failed
echo ERROR: One or both services did not become ready. See the service windows above.
:failed
echo.
pause
exit /b 1
