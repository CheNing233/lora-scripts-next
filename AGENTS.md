# Anima Trainer

这是一个适用于 Anima 进行训练的训练器，进行个人向的实验和代码调整，适应个人向需求

## 文档相关

原始README可查看：
- README-zh.md
- README.md

Anima、LoRA相关可查看：
- docs/* 

## 模型来源

为了避免大文件复制迁移，避免占据额外磁盘空间，统一使用：
```bat
set "MODELS_ROOT=C:\Users\xChenNing\Documents\__AI_MODELS__"

set "MODEL_DIT=%MODELS_ROOT%\unet\anima-base-v1.0.safetensors"
set "MODEL_QWEN3=%MODELS_ROOT%\text_encoders\qwen_3_06b_base.safetensors"
set "MODEL_VAE=%MODELS_ROOT%\vae\qwen_image_vae.safetensors"
```

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
