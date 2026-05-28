from __future__ import annotations

import argparse
import json
from pathlib import Path

from hsb_eeg2text.config import deep_get, load_config
from hsb_eeg2text.datasets.zuco_dataset import ZuCoWordDataset, collate_batch
from hsb_eeg2text.models.model import HierarchicalSemanticModel
from hsb_eeg2text.models.text_encoder import build_text_encoder
from hsb_eeg2text.taxonomy.tree import Taxonomy
from hsb_eeg2text.utils.io import ensure_dir, write_jsonl


def get_table_path(base: Path, name: str) -> Path:
    parquet = base / f"{name}.parquet"
    return parquet if parquet.exists() else base / f"{name}.csv"


def decode(config: dict, checkpoint_path: str | Path, split: str = "test", output_path: str | Path | None = None, smoke: bool = False) -> dict:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_config = checkpoint.get("config", config)
    processed_dir = Path(deep_get(ckpt_config, "paths.processed_zuco_dir"))
    taxonomy_path = Path(deep_get(ckpt_config, "paths.taxonomy_dir")) / "keyword_taxonomy.csv"
    dataset = ZuCoWordDataset(get_table_path(processed_dir, "word_samples"), taxonomy_path, split=split)
    taxonomy = Taxonomy.from_csv(taxonomy_path)
    embedding_dim = int(checkpoint["embedding_dim"])
    model = HierarchicalSemanticModel(
        deep_get(ckpt_config, "model.eeg_encoder", "deep4"),
        embedding_dim,
        float(deep_get(ckpt_config, "model.dropout", 0.2)),
        checkpoint["label_counts"],
        in_channels=int(deep_get(ckpt_config, "data.eeg_shape", [48, 5, 10])[0]),
        text_input_dim=int(checkpoint.get("text_input_dim", embedding_dim)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() and deep_get(ckpt_config, "training.device", "cuda") == "cuda" else "cpu")
    model.to(device).eval()

    text_encoder = build_text_encoder(ckpt_config, embedding_dim, smoke=smoke)
    label_texts = checkpoint["label_texts"]
    text_tables = {}
    with torch.no_grad():
        for level, texts in label_texts.items():
            raw_table = torch.tensor(text_encoder.encode(texts), dtype=torch.float32, device=device)
            text_tables[level] = model.project_text(raw_table)
    label_by_id = {level: {idx: label for label, idx in checkpoint["label_maps"][level].items()} for level in checkpoint["label_maps"]}
    top_k = int(deep_get(ckpt_config, "inference.top_k", 5))
    temperature = float(deep_get(ckpt_config, "training.temperature", 0.07))
    rows = []
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["eeg"].to(device))
            emb = F.normalize(outputs["embedding"], dim=-1)
            scores = {level: emb @ table.T for level, table in text_tables.items()}
            probs = {level: torch.softmax(score / temperature, dim=-1) for level, score in scores.items()}
            for i in range(emb.shape[0]):
                coarse_top = top_indices(probs["coarse"][i], top_k)
                paths = []
                for coarse_idx, coarse_score in coarse_top:
                    coarse = label_by_id["coarse"][coarse_idx]
                    mids = taxonomy.children_of_coarse(coarse)
                    for mid in mids:
                        mid_idx = checkpoint["label_maps"]["mid"][mid]
                        mid_score = float(probs["mid"][i, mid_idx].detach().cpu())
                        for fine in taxonomy.children_of_mid(mid):
                            fine_idx = checkpoint["label_maps"]["fine"][fine]
                            fine_score = float(probs["fine"][i, fine_idx].detach().cpu())
                            paths.append(
                                {
                                    "coarse": coarse,
                                    "mid": mid,
                                    "fine": fine,
                                    "score": float(coarse_score) * mid_score * fine_score,
                                    "fine_score": fine_score,
                                }
                            )
                paths = sorted(paths, key=lambda item: item["score"], reverse=True)[:top_k]
                rows.append(
                    {
                        "sample_id": batch["sample_id"][i],
                        "sentence_id": batch["sentence_id"][i],
                        "word_id": batch["word_id"][i],
                        "word": batch["word"][i],
                        "sentence": batch["sentence"][i],
                        "gold": {"coarse": batch["coarse"][i], "mid": batch["mid"][i], "fine": batch["fine"][i]},
                        "predictions": paths,
                    }
                )
    if output_path is None:
        output_path = Path(deep_get(ckpt_config, "paths.decoded_dir")) / f"decoded_{split}.jsonl"
    write_jsonl(rows, output_path)
    return {"decoded_path": str(output_path), "rows": len(rows)}


def top_indices(vector, k: int) -> list[tuple[int, float]]:
    values, indices = vector.topk(min(k, vector.shape[0]))
    return [(int(idx.detach().cpu()), float(val.detach().cpu())) for idx, val in zip(indices, values)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/zuco_mvp.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(decode(load_config(args.config), args.checkpoint, args.split, args.output, args.smoke), indent=2))


if __name__ == "__main__":
    main()
