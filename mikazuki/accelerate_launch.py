"""Training subprocess entry: enable HF mirror fallback, then run accelerate launch."""

from __future__ import annotations

import sys

import mikazuki.hf_mirror_bootstrap  # noqa: F401 — patch hf_hub_download before sd-scripts imports

from accelerate.commands.launch import main


if __name__ == "__main__":
    sys.argv[0] = "accelerate-launch"
    main()
