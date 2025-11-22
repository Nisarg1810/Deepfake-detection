@echo off
echo ============================================================
echo DEEPFAKE DETECTION - REACT + FLASK INTEGRATION
echo ============================================================
echo.
echo Starting Backend Server (Flask)...
echo.

cd web
start cmd /k "python server.py"

timeout /t 3 /nobreak >nul

echo.
echo Starting Frontend Server (React + Vite)...
echo.

cd ../Frontend
start cmd /k "npm run dev"

echo.
echo ============================================================
echo SERVERS STARTED!
echo ============================================================
echo.
echo Backend:  http://localhost:5000
echo Frontend: http://localhost:5173
echo.
echo Open http://localhost:5173 in your browser
echo.
echo Press any key to exit this window...
echo (The servers will continue running in separate windows)
echo ============================================================
pause >nul
