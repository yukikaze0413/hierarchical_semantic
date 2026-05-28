from __future__ import annotations

import argparse
import json
from pathlib import Path

from hsb_eeg2text.config import deep_get, load_config, save_config
from hsb_eeg2text.datasets.zuco_dataset import ZuCoWordDataset, collate_batch
from hsb_eeg2text.models.losses import hierarchy_consistency_loss, prototype_ce_loss
from hsb_eeg2text.models.model import HierarchicalSemanticModel
from hsb_eeg2text.models.text_encoder import build_text_encoder
from hsb_eeg2text.utils.io import ensure_dir, save_json
from hsb_eeg2text.utils.random import seed_everything


def get_table_path(base: Path, name: str) -> Path:
    parquet = base / f"{name}.parquet"
    return parquet if parquet.exists() else base / f"{name}.csv"


def train(
    config: dict,
    smoke: bool = False,
    epochs_override: int | None = None,
    experiment_name: str | None = None,
    variant: str = "hierarchical",
    shuffle_labels: bool = False,
) -> dict:
    import torch
    from torch.utils.data import DataLoader

    seed_everything(int(deep_get(config, "project.seed", 42)))
    processed_dir = Path(deep_get(config, "paths.processed_zuco_dir"))
    taxonomy_path = Path(deep_get(config, "paths.taxonomy_dir")) / "keyword_taxonomy.csv"
    word_samples_path = get_table_path(processed_dir, "word_samples")
    train_ds = ZuCoWordDataset(word_samples_path, taxonomy_path, split="train")
    val_ds = ZuCoWordDataset(word_samples_path, taxonomy_path, split="val")
    if shuffle_labels:
        shuffle_dataset_labels(train_ds, int(deep_get(config, "project.seed", 42)))
    if len(train_ds) == 0:
        raise ValueError("Training split has no taxonomy-covered samples. Check vocabulary_size, splits, and taxonomy annotation.")
    label_counts = {name: len(mapping) for name, mapping in train_ds.label_maps.items()}
    embedding_dim = int(deep_get(config, "model.embedding_dim", 256))
    in_channels = int(deep_get(config, "data.eeg_shape", [48, 5, 10])[0])
    text_encoder = build_text_encoder(config, embedding_dim, smoke=smoke)
    text_input_dim = int(getattr(text_encoder, "output_dim", embedding_dim))
    model = HierarchicalSemanticModel(
        deep_get(config, "model.eeg_encoder", "deep4"),
        embedding_dim,
        float(deep_get(config, "model.dropout", 0.2)),
        label_counts,
        in_channels=in_channels,
        text_input_dim=text_input_dim,
    )
    device_name = "cuda" if deep_get(config, "training.device", "cuda") == "cuda" and torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model.to(device)

    label_texts = {
        level: [label for label, _ in sorted(mapping.items(), key=lambda item: item[1])]
        for level, mapping in train_ds.label_maps.items()
    }
    text_tables = {
        level: torch.tensor(text_encoder.encode(texts), dtype=torch.float32, device=device)
        for level, texts in label_texts.items()
    }

    batch_size = 8 if smoke else int(deep_get(config, "training.batch_size", 64))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch) if len(val_ds) else None
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(deep_get(config, "training.learning_rate", 3e-4)),
        weight_decay=float(deep_get(config, "training.weight_decay", 1e-4)),
    )

    epochs = epochs_override or (1 if smoke else int(deep_get(config, "training.epochs", 60)))
    temp = float(deep_get(config, "training.temperature", 0.07))
    weights = deep_get(config, "training.loss_weights")
    coarse_mid_until = int(deep_get(config, "training.curriculum.coarse_mid_until_epoch", 20))
    add_fine_until = int(deep_get(config, "training.curriculum.add_fine_until_epoch", 50))
    if variant == "no_curriculum":
        coarse_mid_until = 0
        add_fine_until = 0
    exp_name = experiment_name or ("smoke" if smoke else f"{deep_get(config, 'training.experiment_name', 'zuco_mvp')}_{variant}")
    run_dir = ensure_dir(Path(deep_get(config, "paths.checkpoints_dir")) / exp_name)
    save_config(config, run_dir / "config.yaml")
    save_json(
        {"label_maps": train_ds.label_maps, "label_texts": label_texts, "coverage": {"train": train_ds.coverage, "val": val_ds.coverage}},
        run_dir / "labels.json",
    )

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch in train_loader:
            eeg = batch["eeg"].to(device)
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            outputs = model(eeg)
            coarse_emb = model.project_text(text_tables["coarse"])
            mid_emb = model.project_text(text_tables["mid"])
            fine_emb = model.project_text(text_tables["fine"])
            loss = outputs["embedding"].sum() * 0.0
            if variant != "fine_only":
                loss = loss + weights["coarse"] * prototype_ce_loss(outputs["embedding"], coarse_emb, batch["coarse_id"], temp)
                loss = loss + weights["mid"] * prototype_ce_loss(outputs["embedding"], mid_emb, batch["mid_id"], temp)
            if variant == "fine_only" or epoch > coarse_mid_until:
                loss = loss + weights["fine"] * prototype_ce_loss(outputs["embedding"], fine_emb, batch["fine_id"], temp)
            if variant not in {"fine_only", "no_hierarchy_loss"} and epoch > add_fine_until:
                loss = loss + weights["hierarchy"] * hierarchy_consistency_loss(outputs, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * eeg.shape[0]
            total_items += eeg.shape[0]
        metrics = {"epoch": epoch, "train_loss": total_loss / max(total_items, 1)}
        if val_loader:
            metrics["val_loss"] = evaluate_loss(model, val_loader, text_tables, device, temp, weights, epoch, coarse_mid_until, add_fine_until)
        history.append(metrics)
        print(metrics)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "label_maps": train_ds.label_maps,
        "label_texts": label_texts,
        "embedding_dim": embedding_dim,
        "text_input_dim": text_input_dim,
        "label_counts": label_counts,
        "variant": variant,
        "shuffle_labels": shuffle_labels,
    }
    ckpt_path = run_dir / "latest.pt"
    torch.save(checkpoint, ckpt_path)
    save_json({"history": history, "checkpoint": str(ckpt_path)}, run_dir / "train_metrics.json")
    return {"checkpoint": str(ckpt_path), "run_dir": str(run_dir), "history": history}


def shuffle_dataset_labels(dataset, seed: int) -> None:
    import numpy as np

    rng = np.random.default_rng(seed)
    for col in ["coarse", "mid", "fine"]:
        values = dataset.samples[col].to_numpy().copy()
        rng.shuffle(values)
        dataset.samples[col] = values


def evaluate_loss(model, loader, text_tables, device, temp, weights, epoch: int, coarse_mid_until: int, add_fine_until: int) -> float:
    import torch

    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device)
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            outputs = model(eeg)
            loss = weights["coarse"] * prototype_ce_loss(outputs["embedding"], model.project_text(text_tables["coarse"]), batch["coarse_id"], temp)
            loss = loss + weights["mid"] * prototype_ce_loss(outputs["embedding"], model.project_text(text_tables["mid"]), batch["mid_id"], temp)
            if epoch > coarse_mid_until:
                loss = loss + weights["fine"] * prototype_ce_loss(outputs["embedding"], model.project_text(text_tables["fine"]), batch["fine_id"], temp)
            if epoch > add_fine_until:
                loss = loss + weights["hierarchy"] * hierarchy_consistency_loss(outputs, batch)
            total += float(loss.detach().cpu()) * eeg.shape[0]
            count += eeg.shape[0]
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/zuco_mvp.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--experiment-name")
    parser.add_argument("--variant", default="hierarchical", choices=["hierarchical", "fine_only", "no_hierarchy_loss", "no_curriculum"])
    parser.add_argument("--shuffle-labels", action="store_true")
    args = parser.parse_args()
    result = train(
        load_config(args.config),
        smoke=args.smoke,
        epochs_override=args.epochs,
        experiment_name=args.experiment_name,
        variant=args.variant,
        shuffle_labels=args.shuffle_labels,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
