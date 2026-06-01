@echo off
REM Open the FSM viewer via a local HTTP server so it can auto-fetch
REM state-catalog.csv / transition-table.csv / logs (file:// blocks fetch).
REM Double-click this file. Close the window to stop the server.
cd /d "%~dp0"
start "" "http://localhost:8777/viewer.html"
echo.
echo   FSM viewer: http://localhost:8777/viewer.html
echo   (Close this window to stop the server.)
echo.
python -m http.server 8777
