@echo off
REM Windows 11 demo launcher. Double-click, or run from a terminal.
REM
REM   demo.bat            live webcam
REM   demo.bat clip       the bundled 3-person test clip
REM   demo.bat <path>     any video file
REM
REM Press q or Esc in the preview window to quit. A journey summary prints on
REM exit, and output\sightings.jsonl holds one record per person seen.
REM
REM Nothing here can move the rover -- it drives itself. This only observes.
REM
REM --confirm-min-interval 0.5 roughly doubles FPS versus the PRD default by
REM rate-limiting the expensive confirm tier. Better for a live demo; drop the
REM flag if you want maximum detection precision instead.

setlocal
set PY=%~dp0.venv\Scripts\python.exe

if not exist "%PY%" (
    echo ERROR: .venv not found. Run this first:
    echo     uv sync
    exit /b 1
)

set SOURCE=%1
if "%SOURCE%"=="" set SOURCE=0
if "%SOURCE%"=="clip" set SOURCE=tests\fixtures\pan_clip.mp4

echo Running rescue_vision on source: %SOURCE%
echo Press q or Esc in the preview window to quit.
echo.

"%PY%" -m rescue_vision --source %SOURCE% --display --confirm-min-interval 0.5

endlocal
