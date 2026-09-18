@echo off
REM Windows one-shot setup + demo (CPU only)
if not exist .venv (python -m venv .venv)
call .venv\Scripts\activate
pip install -r requirements-demo.txt
if not exist demo_repos\leon (git clone --depth 60 https://github.com/leon-ai/leon demo_repos\leon)
python scripts\fetch_cache.py
python app.py %*
