@echo off
echo ================================
echo python model API - development
echo port: 8001
echo ================================
echo.

REM activate virtual environment
call venv\Scripts\activate

echo virtual environment activated
echo starting server...
echo.

REM run server with message
uvicorn api.inference_api:app --host 0.0.0.0 --port 8001 --reload

echo.
echo ================================
echo python model API stopped
echo ================================
pause
