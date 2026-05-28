from pathlib import Path

import pytest


def test_dataset_filters_oov_taxonomy_rows(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("torch")

    from hsb_eeg2text.datasets.zuco_dataset import ZuCoWordDataset

    import numpy as np

    eeg_dir = tmp_path / "eeg"
    eeg_dir.mkdir()
    for sample_id in ["in_vocab", "oov"]:
        np.save(eeg_dir / f"{sample_id}.npy", np.zeros((48, 5, 10), dtype=np.float32))

    word_samples = tmp_path / "word_samples.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "in_vocab",
                "subject_id": "S",
                "sentence_id": "s1",
                "word_id": 0,
                "word": "ambulance",
                "lemma": "ambulance",
                "sentence": "The ambulance stopped.",
                "eeg_path": str(eeg_dir / "in_vocab.npy"),
                "coarse": "object",
                "mid": "vehicle",
                "fine": "ambulance",
                "split": "train",
            },
            {
                "sample_id": "oov",
                "subject_id": "S",
                "sentence_id": "s1",
                "word_id": 1,
                "word": "unknown",
                "lemma": "unknown",
                "sentence": "The ambulance stopped.",
                "eeg_path": str(eeg_dir / "oov.npy"),
                "coarse": "object",
                "mid": "misc_object",
                "fine": "unknown",
                "split": "train",
            },
        ]
    ).to_csv(word_samples, index=False)
    taxonomy = tmp_path / "keyword_taxonomy.csv"
    pd.DataFrame(
        [{"keyword": "ambulance", "coarse": "object", "mid": "vehicle", "fine": "ambulance", "frequency": 1}]
    ).to_csv(taxonomy, index=False)

    dataset = ZuCoWordDataset(word_samples, taxonomy, split="train")
    assert len(dataset) == 1
    assert dataset.coverage["rows_dropped_oov"] == 1
