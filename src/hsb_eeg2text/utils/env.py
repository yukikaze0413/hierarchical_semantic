from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from hsb_eeg2text.config import load_config
from hsb_eeg2text.utils.io import save_json


def command_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=15)
    except Exception as exc:
        return f"unavailable: {exc}"


def audit_environment(config_path: str = "configs/zuco_mvp.yaml") -> dict:
    cfg = load_config(config_path)
    total, used, free = shutil.disk_usage(Path.cwd())
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "disk_gb": {
            "total": round(total / 1e9, 2),
            "used": round(used / 1e9, 2),
            "free": round(free / 1e9, 2),
        },
        "nvidia_smi": command_output(["nvidia-smi"]),
        "tmux": shutil.which("tmux"),
        "conda": shutil.which("conda"),
        "config": cfg.get("_config_path"),
        "packages": {},
    }
    for package in ["numpy", "pandas", "torch", "mne", "sentence_transformers", "transformers", "openai"]:
        try:
            module = __import__(package)
            report["packages"][package] = getattr(module, "__version__", "installed")
        except Exception as exc:
            report["packages"][package] = f"missing: {exc}"
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/zuco_mvp.yaml")
    parser.add_argument("--output", default="outputs/reports/env_audit.json")
    args = parser.parse_args()
    report = audit_environment(args.config)
    save_json(report, args.output)
    print(f"Environment audit saved to {args.output}")
    print(f"Disk free: {report['disk_gb']['free']} GB")
    print(report["nvidia_smi"])


if __name__ == "__main__":
    main()
