"""Tests for portable SD-Trainer data directory junction helper."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "portable"))

from link_portable_data_dirs import (  # noqa: E402
    is_portable_layout,
    link_all_portable_data_dirs,
    link_portable_data_dir,
    resolve_portable_roots,
)


@pytest.mark.skipif(os.name != "nt", reason="junctions are Windows-only")
def test_link_portable_data_dir_creates_junction(tmp_path: Path):
    portable_root = tmp_path / "PortableRoot"
    trainer = portable_root / "SD-Trainer"
    trainer.mkdir(parents=True)
    (trainer / "gui.py").write_text("# test\n", encoding="utf-8")
    (portable_root / "python_embeded").mkdir()
    (portable_root / "sd-models").mkdir(parents=True)
    (portable_root / "sd-models" / "marker.txt").write_text("ok", encoding="utf-8")

    result = link_portable_data_dir(trainer, portable_root, "sd-models", log=lambda *_: None)
    assert result == "linked"
    assert (trainer / "sd-models" / "marker.txt").read_text(encoding="utf-8") == "ok"


def test_is_portable_layout_detects_embedded_python(tmp_path: Path):
    portable_root = tmp_path / "PortableRoot"
    trainer = portable_root / "SD-Trainer"
    trainer.mkdir(parents=True)
    (trainer / "gui.py").write_text("# test\n", encoding="utf-8")
    (portable_root / "python_embeded").mkdir()
    assert is_portable_layout(trainer, portable_root) is True


def test_resolve_portable_roots_from_trainer_dir(tmp_path: Path):
    trainer = tmp_path / "SD-Trainer"
    trainer.mkdir()
    resolved_trainer, portable_root = resolve_portable_roots(trainer)
    assert resolved_trainer == trainer.resolve()
    assert portable_root == tmp_path.resolve()


def test_launcher_invokes_link_script():
    launcher = (ROOT / "scripts" / "portable" / "launch_portable.bat").read_text(
        encoding="utf-8"
    )
    assert "link_portable_data_dirs.py" in launcher


def test_build_script_invokes_link_helper():
    script = (ROOT / "build-scripts" / "build_portable.ps1").read_text(encoding="utf-8")
    assert "link_portable_data_dirs.py" in script
