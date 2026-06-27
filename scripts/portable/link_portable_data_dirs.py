"""Create directory junctions inside SD-Trainer/ -> portable root data dirs.

Portable layout keeps user data at <PortableRoot>/{sd-models,output,...} while
gui.py runs with cwd=<PortableRoot>/SD-Trainer. The built-in file picker lists
./sd-models and ./output relative to cwd, so we junction those names into the
trainer directory on Windows portable builds.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PORTABLE_DATA_DIR_NAMES = (
    "sd-models",
    "output",
    "logs",
    "train",
    "tagger-models",
)

FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def resolve_portable_roots(trainer_dir: Path | None = None) -> tuple[Path, Path]:
    if trainer_dir is None:
        trainer_dir = Path.cwd()
    trainer_dir = trainer_dir.resolve()
    portable_root = trainer_dir.parent
    return trainer_dir, portable_root


def is_portable_layout(trainer_dir: Path, portable_root: Path) -> bool:
    if os.name != "nt":
        return False
    if (trainer_dir / "gui.py").is_file() and (portable_root / "python_embeded").is_dir():
        return True
    if (trainer_dir / "PORTABLE_BUILD").is_file():
        return True
    return False


def _is_reparse_point(path: Path) -> bool:
    try:
        attrs = path.lstat().st_file_attributes
    except (AttributeError, OSError):
        return path.is_symlink()
    return bool(attrs & FILE_ATTRIBUTE_REPARSE_POINT)


def _junction_target(link: Path) -> Path | None:
    """Best-effort resolve of a directory junction target on Windows."""
    if not _is_reparse_point(link):
        return None
    try:
        return link.resolve()
    except OSError:
        return None


def _create_junction(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        check=True,
        capture_output=True,
        text=True,
    )


def link_portable_data_dir(
    trainer_dir: Path,
    portable_root: Path,
    name: str,
    *,
    log=print,
) -> str:
    """
    Ensure trainer_dir/name junctions to portable_root/name.

    Returns: linked | skipped | replaced-empty | kept-nonempty
    """
    target = portable_root / name
    link = trainer_dir / name
    target.mkdir(parents=True, exist_ok=True)

    if link.exists() or link.is_symlink():
        if _is_reparse_point(link):
            existing = _junction_target(link)
            if existing is not None and existing.resolve() == target.resolve():
                return "skipped"
            log(f"[portable] {name}: junction exists -> {existing}; expected {target}")
            return "skipped"

        if link.is_dir():
            entries = list(link.iterdir())
            if entries:
                log(
                    f"[portable] {name}: keep existing folder inside SD-Trainer "
                    f"({len(entries)} item(s)); file picker uses this copy, not {target}"
                )
                return "kept-nonempty"
            shutil.rmtree(link)
            _create_junction(link, target)
            log(f"[portable] linked {link} -> {target} (replaced empty folder)")
            return "replaced-empty"

        log(f"[portable] {name}: skip non-directory path {link}")
        return "skipped"

    _create_junction(link, target)
    log(f"[portable] linked {link} -> {target}")
    return "linked"


def link_all_portable_data_dirs(
    trainer_dir: Path | None = None,
    *,
    log=print,
) -> dict[str, str]:
    trainer_dir, portable_root = resolve_portable_roots(trainer_dir)
    if not is_portable_layout(trainer_dir, portable_root):
        log("[portable] not a portable layout; skip data-dir junctions")
        return {}

    results: dict[str, str] = {}
    for name in PORTABLE_DATA_DIR_NAMES:
        try:
            results[name] = link_portable_data_dir(
                trainer_dir,
                portable_root,
                name,
                log=log,
            )
        except subprocess.CalledProcessError as exc:
            err = (exc.stderr or exc.stdout or str(exc)).strip()
            log(f"[portable] failed to link {name}: {err}")
            results[name] = "failed"
        except OSError as exc:
            log(f"[portable] failed to link {name}: {exc}")
            results[name] = "failed"
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trainer-dir",
        type=Path,
        default=None,
        help="SD-Trainer directory (default: current working directory)",
    )
    args = parser.parse_args(argv)
    link_all_portable_data_dirs(args.trainer_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
