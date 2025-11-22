@echo off
echo ============================================================
echo DEEPFAKE DETECTION WEB INTERFACE
echo ============================================================
echo.
echo Starting Flask server...
echo.

cd web
start python server.py

timeout /t 3 /nobreak >nul

echo Server started!
echo.
echo Opening web interface in your browser...
echo.

start index.html

echo ============================================================
echo READY TO USE!
echo ============================================================
echo.
echo The web interface should open in your browser automatically.
echo If not, open this file manually: web/index.html
echo.
echo To stop the server, close the Python window.
echo ============================================================
pause
