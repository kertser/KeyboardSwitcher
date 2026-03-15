@echo off
REM ================================================================
REM  KeyboardSwitcher — Release Build & Package Script
REM  Run from the cpp/ directory (or double-click this .bat file).
REM
REM  Prerequisites:
REM    - CMake 3.20+ on PATH
REM    - MinGW-w64 (g++) on PATH
REM    - NSIS on PATH (optional — only needed for the installer)
REM    - ONNX Runtime SDK in cpp/onnxruntime/
REM ================================================================
setlocal enabledelayedexpansion

cd /d "%~dp0"

set BUILD_DIR=cmake-build-release
set DIST_DIR=dist

echo.
echo ============================================================
echo  KeyboardSwitcher — Release Build
echo ============================================================
echo.

REM ── Step 1: Configure ─────────────────────────────────────────
echo [1/5] Configuring CMake (Release, MinGW Makefiles)...
cmake -B %BUILD_DIR% -DCMAKE_BUILD_TYPE=Release -G "MinGW Makefiles"
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: CMake configure failed.
    pause
    exit /b 1
)
echo      OK
echo.

REM ── Step 2: Build ─────────────────────────────────────────────
echo [2/5] Building...
cmake --build %BUILD_DIR% --config Release -j %NUMBER_OF_PROCESSORS%
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Build failed.
    pause
    exit /b 1
)
echo      OK
echo.

REM ── Step 3: Install to dist/ ──────────────────────────────────
echo [3/5] Installing to %DIST_DIR%/...
cmake --install %BUILD_DIR% --prefix %DIST_DIR%
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Install failed.
    pause
    exit /b 1
)
echo      OK
echo.

REM ── Step 4: Package — ZIP (portable) ──────────────────────────
echo [4/5] Packaging ZIP (portable archive)...
cd %BUILD_DIR%
cpack -G ZIP
if %ERRORLEVEL% neq 0 (
    echo.
    echo WARNING: ZIP packaging failed (non-fatal).
) else (
    echo      OK
)
cd ..
echo.

REM ── Step 5: Package — NSIS (installer) ────────────────────────
echo [5/5] Packaging NSIS installer...
set NSIS_EXE=
where makensis >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set NSIS_EXE=makensis
) else if exist "C:\Program Files (x86)\NSIS\makensis.exe" (
    set "NSIS_EXE=C:\Program Files (x86)\NSIS\makensis.exe"
) else if exist "C:\Program Files\NSIS\makensis.exe" (
    set "NSIS_EXE=C:\Program Files\NSIS\makensis.exe"
)

if "!NSIS_EXE!"=="" (
    echo      SKIPPED — NSIS not found on PATH or in Program Files.
    echo      Install from https://nsis.sourceforge.io/Download
) else (
    echo      Found NSIS: !NSIS_EXE!
    cd %BUILD_DIR%
    cpack -G NSIS -D "CPACK_NSIS_MAKENSIS_EXECUTABLE=!NSIS_EXE!"
    if !ERRORLEVEL! neq 0 (
        echo.
        echo WARNING: NSIS packaging failed (non-fatal).
    ) else (
        echo      OK
    )
    cd ..
)
echo.

REM ── Summary ───────────────────────────────────────────────────
echo ============================================================
echo  Build complete!
echo ============================================================
echo.
echo  Standalone files:  %CD%\%DIST_DIR%\
echo  Packages:          %CD%\%BUILD_DIR%\KeyboardSwitcher-*.zip
echo                     %CD%\%BUILD_DIR%\KeyboardSwitcher-*.exe
echo.

pause

