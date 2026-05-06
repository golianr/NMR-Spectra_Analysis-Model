@echo off
cd /d %~dp0

if not exist .venv (
  echo Creating new virtual environment .venv ...
  py -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if not exist nmr_artifacts_fusion.zip (
  echo.
  echo WARNING: nmr_artifacts_fusion.zip is missing.
  echo Put your trained model ZIP next to app.py with this exact filename.
  echo The app will still open, but prediction will be disabled until the ZIP exists.
  echo.
)

python app.py --server-name 127.0.0.1 --server-port 7860
pause
