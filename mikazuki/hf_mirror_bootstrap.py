"""Patch huggingface_hub downloads to fall back to ModelScope when HF mirrors fail."""

from __future__ import annotations

import os
from typing import Any

from mikazuki.tokenizer_cache import MODELSCOPE_TOKENIZER_REPOS

_PATCHED = False


def _truthy(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _modelscope_revision(revision: str | None) -> str:
    if not revision or revision in {"main", "master"}:
        return "master"
    return revision


def _download_via_modelscope(repo_id: str, filename: str, *, revision: str | None) -> str:
    from modelscope.hub.file_download import model_file_download

    ms_repo = MODELSCOPE_TOKENIZER_REPOS.get(repo_id, repo_id)
    path = model_file_download(ms_repo, filename, revision=_modelscope_revision(revision))
    return str(path)


def patch_hf_hub_download() -> None:
    global _PATCHED
    if _PATCHED:
        return
    if not _truthy(os.environ.get("MIKAZUKI_HF_MIRROR_FALLBACK"), default=True):
        return

    import huggingface_hub
    import huggingface_hub.file_download as file_download

    original = file_download.hf_hub_download

    def wrapped(*args: Any, **kwargs: Any) -> str:
        if kwargs.get("local_files_only"):
            return original(*args, **kwargs)
        try:
            return original(*args, **kwargs)
        except Exception as first_error:
            repo_id = kwargs.get("repo_id")
            filename = kwargs.get("filename")
            if repo_id is None and args:
                repo_id = args[0]
            if filename is None and len(args) > 1:
                filename = args[1]
            if not repo_id or not filename:
                raise first_error
            try:
                return _download_via_modelscope(
                    str(repo_id),
                    str(filename),
                    revision=kwargs.get("revision"),
                )
            except Exception:
                raise first_error

    file_download.hf_hub_download = wrapped  # type: ignore[assignment]
    huggingface_hub.hf_hub_download = wrapped  # type: ignore[assignment]
    _PATCHED = True


# Import side-effect for training subprocess entrypoint.
patch_hf_hub_download()
