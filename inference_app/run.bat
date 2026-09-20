@echo off
rem Double-click to start the hotel review web page.
cd /d "%~dp0"
python -c "import streamlit" 2>nul || pip install -r requirements.txt
python run.py
pause
