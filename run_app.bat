@echo off
if exist venv\Scripts\activate (call venv\Scripts\activate) else (call .venv\Scripts\activate)
streamlit run app.py
pause
