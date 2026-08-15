@echo off
chcp 65001 >nul 2>&1
setlocal

REM ========== run ==========
set "RUN_NAME=doma_ogpt_v1"
set "OUTPUT_NAME=doma_ogpt_v1"

REM ========== paths ==========
set "TRAINER_ROOT=%~dp0"
set "TRAIN_PATH=%TRAINER_ROOT%__TRAIN__\%RUN_NAME%"

REM ========== Anima base models (__AI_MODELS__) ==========
set "MODELS_ROOT=C:\Users\xChenNing\Documents\__AI_MODELS__"
set "MODEL_DIT=%MODELS_ROOT%\unet\anima-base-v1.0.safetensors"
set "MODEL_QWEN3=%MODELS_ROOT%\text_encoders\qwen_3_06b_base.safetensors"
set "MODEL_VAE=%MODELS_ROOT%\vae\qwen_image_vae.safetensors"

cd /d "%TRAINER_ROOT%"
call venv\Scripts\activate

set "HF_HOME=%TRAINER_ROOT%huggingface"

accelerate launch ^
  --num_cpu_threads_per_process 16 ^
  --mixed_precision bf16 ^
  scripts\dev\anima_train_network.py ^
  --config_file "%TRAIN_PATH%\config.toml" ^
  --dataset_config "%TRAIN_PATH%\ds-config.toml" ^
  --pretrained_model_name_or_path "%MODEL_DIT%" ^
  --qwen3 "%MODEL_QWEN3%" ^
  --vae "%MODEL_VAE%" ^
  --output_dir "%TRAIN_PATH%\output" ^
  --output_name "%OUTPUT_NAME%" ^
  --logging_dir "%TRAIN_PATH%\logs"

endlocal
