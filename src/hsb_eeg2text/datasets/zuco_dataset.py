from __future__ import annotations

from pathlib import Path

import numpy as np

from hsb_eeg2text.taxonomy.tree import Taxonomy
from hsb_eeg2text.utils.io import read_table

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - dependency check path
    class Dataset:  # type: ignore
        pass


def build_label_maps(taxonomy: Taxonomy) -> dict[str, dict[str, int]]:
    return {
        "coarse": {label: idx for idx, label in enumerate(taxonomy.coarse_labels)},
        "mid": {label: idx for idx, label in enumerate(taxonomy.mid_labels)},
        "fine": {label: idx for idx, label in enumerate(taxonomy.fine_labels)},
    }


class ZuCoWordDataset(Dataset):
    def __init__(self, word_samples_path: str | Path, taxonomy_path: str | Path, split: str = "train"):
        import torch

        self.samples = read_table(word_samples_path)
        if split != "all" and "split" in self.samples.columns:
            self.samples = self.samples[self.samples["split"] == split].reset_index(drop=True)
        self.taxonomy = Taxonomy.from_csv(taxonomy_path)
        self.label_maps = build_label_maps(self.taxonomy)
        before = len(self.samples)
        required = {"coarse", "mid", "fine"}
        missing = required - set(self.samples.columns)
        if missing:
            raise ValueError(
                f"Word samples are missing semantic columns {sorted(missing)}. "
                "Run taxonomy construction to annotate word_samples before training."
            )
        self.samples = self.samples[self.samples["fine"].astype(str).isin(self.taxonomy.fine_labels)].reset_index(drop=True)
        self.coverage = {
            "rows_before_vocab_filter": before,
            "rows_after_vocab_filter": len(self.samples),
            "rows_dropped_oov": before - len(self.samples),
        }
        self._torch = torch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        row = self.samples.iloc[idx]
        eeg = np.load(row["eeg_path"]).astype(np.float32)
        coarse = str(row["coarse"])
        mid = str(row["mid"])
        fine = str(row["fine"])
        return {
            "sample_id": str(row["sample_id"]),
            "subject_id": str(row["subject_id"]),
            "sentence_id": str(row["sentence_id"]),
            "word_id": int(row["word_id"]),
            "sentence": str(row["sentence"]),
            "word": str(row["word"]),
            "coarse": coarse,
            "mid": mid,
            "fine": fine,
            "eeg": self._torch.from_numpy(eeg),
            "coarse_id": self._torch.tensor(self.label_maps["coarse"][coarse], dtype=self._torch.long),
            "mid_id": self._torch.tensor(self.label_maps["mid"][mid], dtype=self._torch.long),
            "fine_id": self._torch.tensor(self.label_maps["fine"][fine], dtype=self._torch.long),
        }


def collate_batch(batch: list[dict]) -> dict:
    import torch

    out = {key: [item[key] for item in batch] for key in batch[0] if key != "eeg"}
    out["eeg"] = torch.stack([item["eeg"] for item in batch])
    for key in ["coarse_id", "mid_id", "fine_id"]:
        out[key] = torch.stack([item[key] for item in batch])
    return out
