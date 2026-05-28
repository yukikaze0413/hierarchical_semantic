from __future__ import annotations

import argparse
import json
from pathlib import Path

from hsb_eeg2text.config import deep_get, load_config
from hsb_eeg2text.evaluation.metrics import decode_metrics, reconstruction_metrics
from hsb_eeg2text.utils.io import read_jsonl, save_json


def evaluate(config: dict, decoded_path: str | Path | None = None, reconstructed_path: str | Path | None = None, output_path: str | Path | None = None) -> dict:
    results = {}
    if decoded_path:
        results["decode"] = decode_metrics(read_jsonl(decoded_path))
    if reconstructed_path:
        results["reconstruction"] = reconstruction_metrics(read_jsonl(reconstructed_path))
    if output_path is None:
        output_path = Path(deep_get(config, "paths.reports_dir")) / "metrics.json"
    save_json(results, output_path)
    return {"metrics_path": str(output_path), "metrics": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/zuco_mvp.yaml")
    parser.add_argument("--decoded")
    parser.add_argument("--reconstructed")
    parser.add_argument("--output")
    args = parser.parse_args()
    print(json.dumps(evaluate(load_config(args.config), args.decoded, args.reconstructed, args.output), indent=2))


if __name__ == "__main__":
    main()
