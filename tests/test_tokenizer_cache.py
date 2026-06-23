from pathlib import Path

import pytest

from mikazuki.tokenizer_cache import (
    BUNDLED_TOKENIZER_DIRS,
    TOKENIZER_FILES,
    bundled_tokenizer_cache_dir,
    is_tokenizer_bundle_complete,
    tokenizer_local_dir,
)


def test_tokenizer_local_dir_uses_underscore_folder_names():
    root = Path("/cache")
    assert tokenizer_local_dir(root, "openai/clip-vit-large-patch14") == root / "openai_clip-vit-large-patch14"
    assert (
        tokenizer_local_dir(root, "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
        == root / "laion_CLIP-ViT-bigG-14-laion2B-39B-b160k"
    )


def test_is_tokenizer_bundle_complete_requires_all_files(tmp_path: Path):
    root = tmp_path / "tokenizer-cache"
    for repo_id, folder in BUNDLED_TOKENIZER_DIRS.items():
        local = root / folder
        local.mkdir(parents=True)
        for name in TOKENIZER_FILES:
            (local / name).write_text("x", encoding="utf-8")
    assert is_tokenizer_bundle_complete(root)


def test_bundled_tokenizer_cache_dir_returns_none_when_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "tokenizer-cache"
    root.mkdir()
    monkeypatch.setenv("MIKAZUKI_TOKENIZER_CACHE_DIR", str(root))
    assert bundled_tokenizer_cache_dir() is None


def test_bundled_tokenizer_cache_dir_returns_path_when_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "tokenizer-cache"
    for folder in BUNDLED_TOKENIZER_DIRS.values():
        local = root / folder
        local.mkdir(parents=True)
        for name in TOKENIZER_FILES:
            (local / name).write_text("x", encoding="utf-8")
    monkeypatch.setenv("MIKAZUKI_TOKENIZER_CACHE_DIR", str(root))
    assert bundled_tokenizer_cache_dir() == str(root).replace("\\", "/")


def test_apply_tokenizer_cache_dir_injects_for_sdxl_lora(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from mikazuki.app.api import apply_tokenizer_cache_dir

    root = tmp_path / "tokenizer-cache"
    for folder in BUNDLED_TOKENIZER_DIRS.values():
        local = root / folder
        local.mkdir(parents=True)
        for name in TOKENIZER_FILES:
            (local / name).write_text("x", encoding="utf-8")
    monkeypatch.setenv("MIKAZUKI_TOKENIZER_CACHE_DIR", str(root))

    config: dict = {}
    apply_tokenizer_cache_dir(config, "sdxl-lora")
    assert config["tokenizer_cache_dir"] == str(root).replace("\\", "/")


def test_build_accelerate_train_command_uses_mirror_launch_entry():
    from mikazuki.process import build_accelerate_train_command

    args, env, _ = build_accelerate_train_command(
        trainer_file="./vendor/sd-scripts/sdxl_train_network.py",
        toml_path="config/autosave/test.toml",
    )
    assert "accelerate_launch.py" in args[1]
    assert env.get("PYTHONNOUSERSITE") == "1"
