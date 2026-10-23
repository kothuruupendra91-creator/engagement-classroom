@echo off
:: Student Engagement Analyzer – Windows launcher
:: Double-click this file to start the web app.

cd /d "%~dp0"

echo ============================================
echo  Student Engagement Analyzer
echo ============================================
echo.

:: Check that the venv exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo Please run: python -m venv .venv
    echo Then:       .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo Starting Streamlit dashboard...
echo Open your browser at:  http://localhost:8501
echo.
echo Press Ctrl+C in this window to stop.
echo.

streamlit run app.py --server.headless false --browser.gatherUsageStats false

pause
