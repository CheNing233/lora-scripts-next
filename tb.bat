@echo off
chcp 65001 >nul 2>&1
setlocal

REM Keep in sync with RUN_NAME in start-train-cfg.bat
set "RUN_NAME=agm"

set "TRAINER_ROOT=%~dp0"
set "RUN_PATH=%TRAINER_ROOT%__TRAINS__\%RUN_NAME%"
set "LOG_DIR=%RUN_PATH%\logs"

cd /d "%TRAINER_ROOT%"
call venv\Scripts\activate

tensorboard --logdir="%LOG_DIR%" --host=0.0.0.0 --port=16006

endlocal
