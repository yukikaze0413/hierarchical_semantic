from __future__ import annotations

import argparse
from pathlib import Path

from hsb_eeg2text.config import load_config
from hsb_eeg2text.evaluation.evaluate import evaluate
from hsb_eeg2text.inference.decode import decode
from hsb_eeg2text.inference.reconstruct import reconstruct
from hsb_eeg2text.preprocessing.zuco import create_mock_dataset
from hsb_eeg2text.taxonomy.build import build_taxonomy
from hsb_eeg2text.training.train import train
from hsb_eeg2text.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/zuco_mvp.yaml")
    parser.add_argument("--sentences", type=int, default=24)
    args = parser.parse_args()
    cfg = load_config(args.config)

    print("1/5 creating mock ZuCo-like dataset")
    mock = create_mock_dataset(cfg, sentence_limit=args.sentences)
    print(mock)

    print("2/5 building taxonomy")
    taxonomy = build_taxonomy(cfg, vocab_size=100)
    print(taxonomy)

    print("3/5 training one smoke epoch")
    train_result = train(cfg, smoke=True, epochs_override=1, experiment_name="smoke")
    print(train_result)

    print("4/5 decoding and reconstructing with mock LLM")
    decoded_path = Path("outputs/decoded_anchors/smoke_decoded.jsonl")
    decoded = decode(cfg, train_result["checkpoint"], split="test", output_path=decoded_path, smoke=True)
    print(decoded)
    reconstructed_path = Path("outputs/reconstructed_sentences/smoke_reconstructed.jsonl")
    reconstructed = reconstruct(cfg, decoded_path, backend_name="mock", output_path=reconstructed_path)
    print(reconstructed)

    print("5/5 evaluating")
    metrics = evaluate(cfg, decoded_path, reconstructed_path, output_path="outputs/reports/smoke_metrics.json")
    print(metrics)

    save_json(
        {
            "mock": mock,
            "taxonomy": taxonomy,
            "train": train_result,
            "decoded": decoded,
            "reconstructed": reconstructed,
            "metrics": metrics,
        },
        "outputs/reports/smoke_summary.json",
    )
    print("Smoke test complete: outputs/reports/smoke_summary.json")


if __name__ == "__main__":
    main()
