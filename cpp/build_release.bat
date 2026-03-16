@echo off
REM ================================================================
REM  KeyboardSwitcher — Release Build & Package Script
REM  Run from the cpp/ directory (or double-click this .bat file).
REM
REM  Output:  release/KeyboardSwitcher-<ver>-win64.zip
REM           release/KeyboardSwitcher-<ver>-win64.exe
REM
REM  Prerequisites:
REM    - CMake 3.20+ on PATH
REM    - MinGW-w64 (g++) on PATH
REM    - NSIS on PATH or in Program Files (for the installer)
REM    - ONNX Runtime SDK in cpp/onnxruntime/
REM ================================================================
setlocal enabledelayedexpansion

cd /d "%~dp0"

set BUILD_DIR=cmake-build-release
set RELEASE_DIR=release

echo.
echo ============================================================
echo  KeyboardSwitcher — Release Build
echo ============================================================
echo.

REM ── Step 0: Clean previous build ─────────────────────────────
if exist %BUILD_DIR% (
    echo [0/6] Removing previous build directory...
    rmdir /s /q %BUILD_DIR%
    echo      OK
    echo.
)

REM ── Step 1: Configure ─────────────────────────────────────────
echo [1/6] Configuring CMake (Release, MinGW Makefiles)...
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
echo [2/6] Building...
cmake --build %BUILD_DIR% --config Release -j %NUMBER_OF_PROCESSORS%
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Build failed.
    pause
    exit /b 1
)
echo      OK
echo.

REM ── Step 3: Package — ZIP (portable) ──────────────────────────
echo [3/6] Packaging ZIP (portable archive)...
cd %BUILD_DIR%
cpack -G ZIP
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: ZIP packaging failed.
    cd ..
    pause
    exit /b 1
)
echo      OK
cd ..
echo.

REM ── Step 4: Package — NSIS (installer) ────────────────────────
echo [4/6] Packaging NSIS installer...
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
    echo      ERROR: NSIS not found on PATH or in Program Files.
    echo      Install from https://nsis.sourceforge.io/Download
    echo      Both ZIP and NSIS installer are required for release.
    pause
    exit /b 1
) else (
    echo      Found NSIS: !NSIS_EXE!
    cd %BUILD_DIR%
    cpack -G NSIS -D "CPACK_NSIS_MAKENSIS_EXECUTABLE=!NSIS_EXE!"
    if !ERRORLEVEL! neq 0 (
        echo.
        echo ERROR: NSIS packaging failed.
        cd ..
        pause
        exit /b 1
    )
    echo      OK
    cd ..
)
echo.

REM ── Step 5: Collect artifacts into release/ ───────────────────
echo [5/6] Collecting release artifacts...
if not exist %RELEASE_DIR% mkdir %RELEASE_DIR%

set FOUND_ZIP=0
set FOUND_EXE=0

for %%F in (%BUILD_DIR%\KeyboardSwitcher-*-win64.zip) do (
    copy /y "%%F" %RELEASE_DIR%\ >nul
    set FOUND_ZIP=1
    echo      %%~nxF
)
for %%F in (%BUILD_DIR%\KeyboardSwitcher-*-win64.exe) do (
    copy /y "%%F" %RELEASE_DIR%\ >nul
    set FOUND_EXE=1
    echo      %%~nxF
)

if !FOUND_ZIP!==0 (
    echo      ERROR: ZIP package not found in build output.
    pause
    exit /b 1
)
if !FOUND_EXE!==0 (
    echo      ERROR: NSIS installer not found in build output.
    pause
    exit /b 1
)
echo      OK
echo.

REM ── Step 6: Cleanup build directory ───────────────────────────
echo [6/6] Cleaning up build directory...
rmdir /s /q %BUILD_DIR%
echo      OK
echo.

REM ── Summary ───────────────────────────────────────────────────
echo ============================================================
echo  Build complete!
echo ============================================================
echo.
echo  Release artifacts:
for %%F in (%RELEASE_DIR%\KeyboardSwitcher-*.*) do (
    echo      %CD%\%RELEASE_DIR%\%%~nxF
)
echo.

pause
