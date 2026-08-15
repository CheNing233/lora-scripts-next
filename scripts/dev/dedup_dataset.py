#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练集相似度去重工具：支持 VAE / DINOv2 / CLIP 三种编码器对比。

针对场景：背景相同、人物相同，仅动作/衣物不同的图，只保留一张。

用法示例：
  # DINOv2 语义去重（推荐，先看结果不删除）
  python scripts/dev/dedup_dataset.py --dataset_dir "__TRAIN__/doma_v3/datasets" --encoder dino --threshold 0.85

  # 用项目 VAE 做结构近似去重
  python scripts/dev/dedup_dataset.py --dataset_dir "__TRAIN__/doma_v3/datasets" --encoder vae --threshold 0.90 --size 256

  # CLIP
  python scripts/dev/dedup_dataset.py --dataset_dir "__TRAIN__/doma_v3/datasets" --encoder clip --threshold 0.88

默认只输出 JSON，不删除任何文件。确认无误后追加 --apply 才会删除（危险，建议先备份）。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DEFAULT_VAE = r"C:\Users\xChenNing\Documents\__AI_MODELS__\vae\qwen_image_vae.safetensors"


def list_images(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def load_rgb(path: Path, size: int):
    img = Image.open(path).convert("RGB")
    return img.resize((size, size), Image.LANCZOS)


def l2norm(x: torch.Tensor):
    return F.normalize(x, dim=1)


class VaeEncoder:
    """使用项目里的 qwen_image_vae 编码，输出展平后的归一化 latent 向量。"""

    def __init__(self, vae_path: str, device: torch.device):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from library.qwen_image_autoencoder_kl import load_vae

        self.vae = load_vae(vae_path, device=device)
        self.device = device

    def encode(self, paths, size: int, batch: int) -> torch.Tensor:
        feats = []
        for i in range(0, len(paths), batch):
            tensors = []
            for p in paths[i : i + batch]:
                img = load_rgb(p, size)
                t = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1
                tensors.append(t)
            x = torch.cat(tensors, 0).to(self.vae.dtype).to(self.device)
            with torch.no_grad():
                lat = self.vae.encode_pixels_to_latents(x)  # [B, 16, H/8, W/8]
            feats.append(lat.flatten(1).cpu())
        return l2norm(torch.cat(feats, 0).float())


class DinoEncoder:
    """DINOv2 语义编码，对姿态/动作/背景更鲁棒，适合“同角色不同动作”。"""

    def __init__(self, model_id: str, device: torch.device):
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.device = device

    def encode(self, paths, size: int, batch: int) -> torch.Tensor:
        feats = []
        for i in range(0, len(paths), batch):
            imgs = [load_rgb(p, size) for p in paths[i : i + batch]]
            inputs = self.processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs)
            h = out.last_hidden_state  # [B, N, dim]
            f = h[:, 1:, :].mean(dim=1)  # 去掉 CLS，patch 平均
            feats.append(f.cpu())
        return l2norm(torch.cat(feats, 0).float())


class ClipEncoder:
    """OpenCLIP 图像编码。"""

    def __init__(self, model_name: str, device: torch.device):
        import open_clip

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
        self.model = self.model.to(device).eval()
        self.device = device

    def encode(self, paths, size: int, batch: int) -> torch.Tensor:
        feats = []
        with torch.no_grad():
            for i in range(0, len(paths), batch):
                tensors = [self.preprocess(Image.open(p).convert("RGB")).unsqueeze(0) for p in paths[i : i + batch]]
                x = torch.cat(tensors, 0).to(self.device)
                feats.append(self.model.encode_image(x).cpu())
        return l2norm(torch.cat(feats, 0).float())


def build_groups(sim: torch.Tensor, threshold: float):
    """把 cosine 相似度 >= threshold 的图用并查集聚成组。"""
    n = sim.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    idx = torch.nonzero(sim >= threshold, as_tuple=False).cpu().numpy()
    for a, b in idx:
        union(int(a), int(b))

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", required=True, help="训练集目录，递归扫描")
    ap.add_argument("--encoder", choices=["vae", "dino", "clip"], default="dino")
    ap.add_argument("--vae_path", default=DEFAULT_VAE, help="仅 --encoder vae 时使用")
    ap.add_argument("--size", type=int, default=256, help="编码前统一缩放的边长（VAE 需为 8 的倍数）")
    ap.add_argument("--threshold", type=float, default=0.85, help="cosine 相似度阈值，越高越保守")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--json", default="dedup_result.json")
    ap.add_argument("--apply", action="store_true", help="危险：删除每组中除保留文件外的其余文件")
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    paths = list_images(root)
    if not paths:
        raise SystemExit(f"未找到图片: {root}")
    print(f"找到 {len(paths)} 张图片")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.encoder == "vae":
        enc = VaeEncoder(args.vae_path, device)
    elif args.encoder == "dino":
        enc = DinoEncoder("facebook/dinov2-small", device)
    else:
        enc = ClipEncoder("ViT-B-32", device)

    print(f"[{args.encoder}] 编码中...")
    feats = enc.encode(paths, args.size, args.batch)
    feats = feats.to(device)
    sim = feats @ feats.T
    sim.fill_diagonal_(0.0)

    groups = build_groups(sim, args.threshold)
    print(f"阈值 {args.threshold} 下共 {len(groups)} 个重复组")

    result = {"threshold": args.threshold, "encoder": args.encoder, "groups": []}
    for g in groups:
        members = []
        for i in g:
            p = paths[i]
            members.append({"path": str(p), "size": p.stat().st_size})
        # 默认保留文件大小最大的一张，其余视为待删除
        members_sorted = sorted(members, key=lambda m: -m["size"])
        keep = members_sorted[0]
        remove = members_sorted[1:]
        result["groups"].append(
            {"keep": keep["path"], "remove": [m["path"] for m in remove], "members": members}
        )

    out = Path(args.json)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入 {out}")

    total_remove = sum(len(g["remove"]) for g in result["groups"])
    print(f"可删除 {total_remove} 张（每组保留文件最大的一张）")

    if args.apply:
        removed = 0
        for g in result["groups"]:
            for rp in g["remove"]:
                p = Path(rp)
                if p.exists():
                    p.unlink()
                    removed += 1
        print(f"已删除 {removed} 张")
    else:
        print("未删除任何文件（dry-run）。确认无误后追加 --apply 再执行。")


if __name__ == "__main__":
    main()