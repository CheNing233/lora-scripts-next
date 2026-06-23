"""Domestic Hugging Face downloads via ModelScope's official hub patch.

hf-mirror.com only mirrors metadata; /resolve/ file URLs redirect (308) back to
huggingface.co. Setting HF_ENDPOINT=https://modelscope.cn does not work either
because huggingface_hub speaks a different API than ModelScope.

ModelScope ships ``modelscope.utils.hf_util.patch_hub()`` which replaces
``hf_hub_download`` / ``snapshot_download`` so transformers and sd-scripts keep
using Hugging Face repo ids in code while files are fetched from modelscope.cn
and placed in the normal HF cache layout (when ``cache_dir`` is passed).

Some Hugging Face repo ids differ on ModelScope (e.g. openai/clip-vit-large-patch14
→ AI-ModelScope/clip-vit-large-patch14). ``HF_TO_MODELSCOPE_REPOS`` handles that.
"""

from __future__ import annotations

import os
from typing import Any, Callable

_PATCHED = False

# Hugging Face repo id → ModelScope repo id (extend as we validate more assets).
HF_TO_MODELSCOPE_REPOS: dict[str, str] = {
    "openai/clip-vit-large-patch14": "AI-ModelScope/clip-vit-large-patch14",
    # laion/CLIP-ViT-bigG-14-laion2B-39B-b160k exists under the same id on ModelScope.
}

# Back-compat alias used by tokenizer prefetch.
MODELSCOPE_TOKENIZER_REPOS = HF_TO_MODELSCOPE_REPOS


def remap_hf_repo_id(repo_id: str) -> str:
    return HF_TO_MODELSCOPE_REPOS.get(repo_id, repo_id)


def hub_backend() -> str:
    """Return ``modelscope`` or ``huggingface`` for download routing."""
    explicit = (os.environ.get("MIKAZUKI_HUB_BACKEND") or "auto").strip().lower()
    if explicit in {"hf", "huggingface", "direct"}:
        return "huggingface"
    if explicit in {"ms", "modelscope", "魔搭"}:
        return "modelscope"

    endpoint = (os.environ.get("HF_ENDPOINT") or "").strip().lower()
    if "modelscope" in endpoint or "hf-mirror" in endpoint:
        return "modelscope"

    if explicit == "auto":
        try:
            from mikazuki.launch_utils import network_gfw_test

            if not network_gfw_test():
                return "modelscope"
        except Exception:
            return "modelscope"

    return "huggingface"


def _wrap_repo_remap(fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if args:
            args = (remap_hf_repo_id(str(args[0])), *args[1:])
        if kwargs.get("repo_id") is not None:
            kwargs = {**kwargs, "repo_id": remap_hf_repo_id(str(kwargs["repo_id"]))}
        return fn(*args, **kwargs)

    return wrapped


def _apply_repo_id_remapping() -> None:
    import huggingface_hub

    huggingface_hub.hf_hub_download = _wrap_repo_remap(huggingface_hub.hf_hub_download)  # type: ignore[method-assign]
    huggingface_hub.file_download.hf_hub_download = huggingface_hub.hf_hub_download  # type: ignore[attr-defined]
    if hasattr(huggingface_hub, "snapshot_download"):
        huggingface_hub.snapshot_download = _wrap_repo_remap(huggingface_hub.snapshot_download)  # type: ignore[method-assign]

    # patch_hub() may import transformers before we remap; re-bind its cached import.
    try:
        import transformers.utils.hub as transformers_hub

        transformers_hub.hf_hub_download = huggingface_hub.hf_hub_download
        if hasattr(huggingface_hub, "snapshot_download") and hasattr(transformers_hub, "snapshot_download"):
            transformers_hub.snapshot_download = huggingface_hub.snapshot_download
    except ImportError:
        pass


def _patch_modelscope_download_aliases() -> None:
    """Remap HF repo ids inside ModelScope download entrypoints."""
    import modelscope
    import modelscope.hub.file_download as ms_file

    if not getattr(ms_file, "_mikazuki_repo_remap_patched", False):
        original_file = ms_file.model_file_download

        def model_file_download(model_id: str, *args: Any, **kwargs: Any) -> str:
            return original_file(remap_hf_repo_id(model_id), *args, **kwargs)

        ms_file.model_file_download = model_file_download  # type: ignore[assignment]
        ms_file._mikazuki_repo_remap_patched = True  # type: ignore[attr-defined]

    if not getattr(modelscope, "_mikazuki_repo_remap_patched", False):
        original_snapshot = modelscope.snapshot_download

        def snapshot_download(model_id: str, *args: Any, **kwargs: Any) -> str:
            return original_snapshot(remap_hf_repo_id(model_id), *args, **kwargs)

        modelscope.snapshot_download = snapshot_download  # type: ignore[assignment]
        modelscope._mikazuki_repo_remap_patched = True  # type: ignore[attr-defined]


def enable_china_hub(*, force: bool = False) -> bool:
    """Route huggingface_hub downloads through ModelScope when appropriate.

    Safe to call multiple times. Returns True when ModelScope patch is active.
    """
    global _PATCHED
    if _PATCHED:
        return True
    if not force and hub_backend() != "modelscope":
        return False

    try:
        from modelscope.utils.hf_util import patch_hub
    except ImportError:
        return False

    try:
        import diffusers  # noqa: F401
        import peft  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        pass

    _patch_modelscope_download_aliases()
    patch_hub()
    _apply_repo_id_remapping()
    _PATCHED = True
    return True


def china_hub_status() -> dict[str, str | bool]:
    return {
        "backend": hub_backend(),
        "patched": _PATCHED,
        "hf_endpoint": os.environ.get("HF_ENDPOINT") or "",
    }
