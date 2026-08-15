You are a helpful software engineer assistant

# Anima Trainer

这是一个适用于 Anima 进行训练的训练器，进行个人向的实验和代码调整，适应个人向需求

## 文档相关

原始README可查看：
- `README-zh.md`
- `README.md`

Anima、LoRA相关可查看：
- `docs/*`

研究、调研类文档在：
- `__DOCS__/*`
- `__DOCS__/基础实践.md`

## 模型来源

为了避免大文件复制迁移，避免占据额外磁盘空间，统一使用：
```bat
set "MODELS_ROOT=C:\Users\xChenNing\Documents\__AI_MODELS__"

set "MODEL_DIT=%MODELS_ROOT%\unet\anima-base-v1.0.safetensors"
set "MODEL_QWEN3=%MODELS_ROOT%\text_encoders\qwen_3_06b_base.safetensors"
set "MODEL_VAE=%MODELS_ROOT%\vae\qwen_image_vae.safetensors"
```

## sd-scripts 实际来源

本项目 Anima 训练实际使用的 sd-scripts 是 `vendor/sd-scripts`（基于上游 `kohya-ss/sd-scripts`，本地含项目定制）。

- 实际代码路径：`vendor/sd-scripts`
- 上游仓库：`https://github.com/kohya-ss/sd-scripts`
- 锁定提交：`068bcd7ffe76b2cd5012fb680a2c94e295398bbc`（短 hash `068bcd7`，记录在 `config/anima_backend.toml` 的 `pinned_commit`）

训练入口链：
- `start-train-cfg.bat` → `scripts\dev\anima_train_network.py`（本地兼容 wrapper）→ 读取 `config/anima_backend.toml` → 执行 `vendor/sd-scripts/anima_train_network.py`
- `run_gui.ps1` / `run_gui.bat` → `python gui.py` → `mikazuki/app/api.py` 的 `trainer_mapping`；`anima-lora` / `sd3-lora` / `anima-finetune` 指向 `scripts/dev` 下的 wrapper，最终同样落到 `vendor/sd-scripts`

项目里还保留其它 sd-scripts 副本（注意区分）：
- `scripts/dev/`：`COMMIT_ID = 18e62515c49fe502ca31b30ea2214a97a2e99633`（含 Anima wrapper 与 flux 等脚本）
- `scripts/stable/`：`COMMIT_ID = 8f4ee8fc343b047965cd8976fca65c3a35b7593a`（旧版 stable 脚本，GUI 用于 `sd-lora` / `sd-dreambooth` / `sdxl-finetune` 等路由）

Anima 训练的核心实现以 `vendor/sd-scripts` 为准
如果要修改实现或者进行训练，务必确保走的是 `vendor/sd-scripts`

## 训练规范

训练文件夹放置在：
`__TRAIN__\<具体的训练名称>`

例如：`__TRAIN__\doma_v3\*`

其中包含：
- `__TRAIN__\doma_v3\config.toml`，训练配置
- `__TRAIN__\doma_v3\ds-config.toml`，训练集配置
- `__TRAIN__\doma_v3\logs\*`，tensorboard log
- `__TRAIN__\doma_v3\datasets\*`，训练集数据，往下子目录为子集
- `__TRAIN__\doma_v3\output`，LoRA输出

通过`start-train-cfg.bat`拉起训练
tensorboard可通过`tb.bat`拉起到后台

注意`start-train-cfg.bat`和`tb.bat`都要修改，指向`具体的训练名称`（如`doma_v3`）

## 提交规范

避免将训练集、训练输出等LFS或散落的大数量文件上传到Git
