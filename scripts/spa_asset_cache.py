"""Shared SPA dist cache-bust key for in-place patched frontend bundles."""

from __future__ import annotations

# Bump this whenever frontend/dist assets are patched in place (same filename hash).
SPA_ASSET_CACHE_KEY = "20260627-lokr-preview"

# Previous keys replaced by scripts/bump_spa_asset_cache_key.py when bumping.
LEGACY_SPA_ASSET_CACHE_KEYS = (
    "20260605-routefix2",
    "20260627-config-import",
    "20260627-lokr-guard",
    "20260627-lokr-download",
)

IN_PLACE_PATCHED_DIST_ASSETS = (
    "/assets/app.547295de.js",
    "/assets/layout.96d49288.js",
)
