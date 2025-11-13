# eval/hw_profile.py
from __future__ import annotations
import os
import re
import subprocess
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import platform
import psutil  # if you prefer no dep: parse sysctl hw.memsize


def _soc_string() -> str:
    # Prefer sysctl on macOS for precise SoC branding
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            if out:
                return out
        except Exception:
            pass
    return platform.processor() or "unknown"


def _ram_gb() -> int:
    try:
        return int(round(psutil.virtual_memory().total / (1024**3)))
    except Exception:
        return 0


@dataclass
class HWProfile:
    key: str
    config: Dict[str, Any]


def load_profiles(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def match_profile(cfg: Dict[str, Any]) -> HWProfile:
    soc = _soc_string()
    ram = _ram_gb()
    profiles = cfg.get("profiles", {})
    default_key = cfg.get("default_profile")

    for k, v in profiles.items():
        m = v.get("matches", {})
        soc_re = m.get("soc_regex")
        min_ram = int(m.get("min_ram_gb", 0))
        if soc_re and re.search(soc_re, soc) and ram >= min_ram:
            return HWProfile(k, v)

    # fallback
    if default_key and default_key in profiles:
        return HWProfile(default_key, profiles[default_key])
    # top-level fallback (unknown apple silicon)
    if "unknown-apple-silicon" in cfg:
        return HWProfile("unknown-apple-silicon", cfg["unknown-apple-silicon"])
    raise RuntimeError("No matching hardware profile found.")


def require_same_profile(a: str, b: str) -> None:
    if a != b:
        raise SystemExit(
            f"Speed comparison refused: profiles differ (current={a}, baseline={b}). "
            "Pass --allow-cross-hw to override."
        )
