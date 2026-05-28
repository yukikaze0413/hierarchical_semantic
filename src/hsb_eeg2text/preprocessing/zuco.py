from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hsb_eeg2text.config import deep_get, load_config
from hsb_eeg2text.utils.io import ensure_dir, save_json, write_table
from hsb_eeg2text.utils.random import seed_everything


MOCK_SENTENCES = [
    "The ambulance stopped near the university hospital.",
    "A doctor examined the patient in the small clinic.",
    "The student read a difficult article about memory.",
    "The train moved quickly through the central station.",
    "A painful injury changed the athlete's plan.",
    "The teacher explained the complex problem clearly.",
    "The car waited outside the public library.",
    "A nurse carried medicine into the quiet room.",
    "The child watched a bird near the garden.",
    "The scientist measured activity during sleep.",
]

LEXICON = {
    "ambulance": ("object", "vehicle", "ambulance"),
    "hospital": ("place", "medical_facility", "hospital"),
    "doctor": ("person", "medical_professional", "doctor"),
    "patient": ("person", "patient", "patient"),
    "clinic": ("place", "medical_facility", "clinic"),
    "student": ("person", "learner", "student"),
    "article": ("object", "document", "article"),
    "memory": ("attribute", "cognition", "memory"),
    "train": ("object", "vehicle", "train"),
    "station": ("place", "transport_facility", "station"),
    "injury": ("event", "health_event", "injury"),
    "athlete": ("person", "sports_person", "athlete"),
    "teacher": ("person", "education_professional", "teacher"),
    "problem": ("abstract", "task", "problem"),
    "car": ("object", "vehicle", "car"),
    "library": ("place", "institution", "library"),
    "nurse": ("person", "medical_professional", "nurse"),
    "medicine": ("object", "medical_item", "medicine"),
    "room": ("place", "building_part", "room"),
    "child": ("person", "child", "child"),
    "bird": ("object", "animal", "bird"),
    "garden": ("place", "outdoor_area", "garden"),
    "scientist": ("person", "research_professional", "scientist"),
    "activity": ("event", "physiology", "activity"),
    "sleep": ("event", "physiology", "sleep"),
    "stopped": ("action", "motion", "stop"),
    "examined": ("action", "medical_action", "examine"),
    "read": ("action", "communication", "read"),
    "moved": ("action", "motion", "move"),
    "changed": ("action", "change", "change"),
    "explained": ("action", "communication", "explain"),
    "waited": ("action", "motion", "wait"),
    "carried": ("action", "motion", "carry"),
    "watched": ("action", "perception", "watch"),
    "measured": ("action", "measurement", "measure"),
    "university": ("place", "institution", "university"),
    "difficult": ("attribute", "difficulty", "difficult"),
    "quickly": ("attribute", "speed", "quick"),
    "painful": ("attribute", "sensation", "painful"),
    "complex": ("attribute", "difficulty", "complex"),
    "clearly": ("attribute", "clarity", "clear"),
    "small": ("attribute", "size", "small"),
    "central": ("attribute", "location", "central"),
    "public": ("attribute", "access", "public"),
    "quiet": ("attribute", "sound", "quiet"),
}


def tokenize(sentence: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch == "'" else " " for ch in sentence)
    return [tok for tok in cleaned.split() if tok not in {"the", "a", "an", "in", "near", "about", "into", "during", "through", "outside", "of"}]


def audit_raw_zuco(raw_dir: str | Path) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    files = [p for p in raw_dir.rglob("*") if p.is_file()]
    return {
        "raw_dir": str(raw_dir),
        "exists": raw_dir.exists(),
        "file_count": len(files),
        "total_gb": round(sum(p.stat().st_size for p in files) / 1e9, 4),
        "mat_files": [str(p) for p in files if p.suffix.lower() == ".mat"][:50],
        "sample_files": [str(p) for p in files[:50]],
    }


def raw_epoch_to_band_windows(epoch: np.ndarray, bands: dict[str, list[float]], sfreq: float, windows: int) -> np.ndarray:
    """Convert a channels x time raw epoch to channels x bands x windows bandpower."""
    if epoch.ndim != 2:
        raise ValueError(f"Expected 2D epoch [channels,time], got shape {epoch.shape}")
    channels, n_time = epoch.shape
    chunks = np.array_split(np.arange(n_time), windows)
    out = np.zeros((channels, len(bands), windows), dtype=np.float32)
    for wi, idx in enumerate(chunks):
        segment = epoch[:, idx]
        freqs = np.fft.rfftfreq(segment.shape[1], d=1.0 / sfreq)
        spectrum = np.abs(np.fft.rfft(segment, axis=1)) ** 2
        for bi, (_, (lo, hi)) in enumerate(bands.items()):
            mask = (freqs >= lo) & (freqs < hi)
            out[:, bi, wi] = spectrum[:, mask].mean(axis=1) if mask.any() else 0.0
    return out


def split_for_index(index: int, total: int, train_ratio: float, val_ratio: float) -> str:
    frac = index / max(total, 1)
    if frac < train_ratio:
        return "train"
    if frac < train_ratio + val_ratio:
        return "val"
    return "test"


def create_mock_dataset(config: dict[str, Any], sentence_limit: int | None = None) -> dict[str, str]:
    import pandas as pd

    seed_everything(int(deep_get(config, "project.seed", 42)))
    processed_dir = ensure_dir(deep_get(config, "paths.processed_zuco_dir"))
    eeg_dir = ensure_dir(deep_get(config, "paths.eeg_word_dir"))
    shape = tuple(deep_get(config, "data.eeg_shape", [48, 5, 10]))
    total_sentences = sentence_limit or int(deep_get(config, "data.mock_sentences", 64))
    rng = np.random.default_rng(int(deep_get(config, "project.seed", 42)))

    word_rows: list[dict[str, Any]] = []
    sentence_rows: list[dict[str, Any]] = []
    train_ratio = float(deep_get(config, "data.train_ratio", 0.8))
    val_ratio = float(deep_get(config, "data.val_ratio", 0.1))

    for sent_idx in range(total_sentences):
        sentence = MOCK_SENTENCES[sent_idx % len(MOCK_SENTENCES)]
        split = split_for_index(sent_idx, total_sentences, train_ratio, val_ratio)
        anchors = []
        for word_idx, word in enumerate(tokenize(sentence)):
            if word not in LEXICON:
                continue
            coarse, mid, fine = LEXICON[word]
            sample_id = f"mock_s{sent_idx:04d}_w{word_idx:03d}"
            eeg_path = eeg_dir / f"{sample_id}.npy"
            base = (abs(hash((coarse, mid, fine))) % 1000) / 1000.0
            eeg = rng.normal(loc=base, scale=0.5, size=shape).astype(np.float32)
            np.save(eeg_path, eeg)
            row = {
                "sample_id": sample_id,
                "subject_id": "S_MOCK",
                "sentence_id": f"sent_{sent_idx:04d}",
                "word_id": word_idx,
                "word": word,
                "lemma": fine,
                "sentence": sentence,
                "eeg_path": str(eeg_path),
                "coarse": coarse,
                "mid": mid,
                "fine": fine,
                "split": split,
            }
            word_rows.append(row)
            anchors.append([coarse, mid, fine])
        sentence_rows.append(
            {
                "sentence_id": f"sent_{sent_idx:04d}",
                "sentence": sentence,
                "split": split,
                "anchors_json": json.dumps(anchors),
                "fine_keywords_json": json.dumps([a[2] for a in anchors]),
            }
        )

    word_path = write_table(pd.DataFrame(word_rows), processed_dir / "word_samples.parquet")
    sentence_path = write_table(pd.DataFrame(sentence_rows), processed_dir / "sentence_samples.parquet")
    return {"word_samples": str(word_path), "sentence_samples": str(sentence_path), "eeg_dir": str(eeg_dir)}


def preprocess_manifest(config: dict[str, Any], manifest_path: str | Path, sample_limit: int | None = None) -> dict[str, str]:
    import pandas as pd

    manifest = pd.read_csv(manifest_path)
    required = {"subject_id", "sentence_id", "word_id", "word", "lemma", "sentence", "eeg_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    if sample_limit:
        manifest = manifest.head(sample_limit).copy()

    processed_dir = ensure_dir(deep_get(config, "paths.processed_zuco_dir"))
    eeg_dir = ensure_dir(deep_get(config, "paths.eeg_word_dir"))
    bands = deep_get(config, "preprocessing.frequency_bands")
    windows = int(deep_get(config, "preprocessing.time_windows", 10))
    sfreq = float(deep_get(config, "preprocessing.downsample_hz", 250))
    target_shape = tuple(deep_get(config, "data.eeg_shape", [48, 5, 10]))

    rows = []
    for idx, row in manifest.iterrows():
        eeg = np.load(row["eeg_path"]).astype(np.float32)
        if eeg.shape != target_shape:
            eeg = raw_epoch_to_band_windows(eeg, bands, sfreq=sfreq, windows=windows)
        if eeg.shape != target_shape:
            raise ValueError(f"EEG for row {idx} has shape {eeg.shape}, expected {target_shape}")
        sample_id = row.get("sample_id", f"{row.subject_id}_{row.sentence_id}_{int(row.word_id):04d}")
        out_path = eeg_dir / f"{sample_id}.npy"
        np.save(out_path, eeg)
        rows.append({**row.to_dict(), "sample_id": sample_id, "eeg_path": str(out_path)})

    df = pd.DataFrame(rows)
    if "split" not in df.columns:
        sentence_ids = list(dict.fromkeys(df["sentence_id"].tolist()))
        split_map = {
            sid: split_for_index(i, len(sentence_ids), float(deep_get(config, "data.train_ratio", 0.8)), float(deep_get(config, "data.val_ratio", 0.1)))
            for i, sid in enumerate(sentence_ids)
        }
        df["split"] = df["sentence_id"].map(split_map)

    sentence_rows = []
    for sid, sdf in df.groupby("sentence_id", sort=False):
        anchors = sdf[["coarse", "mid", "fine"]].values.tolist() if {"coarse", "mid", "fine"}.issubset(sdf.columns) else []
        sentence_rows.append(
            {
                "sentence_id": sid,
                "sentence": sdf["sentence"].iloc[0],
                "split": sdf["split"].iloc[0],
                "anchors_json": json.dumps(anchors),
                "fine_keywords_json": json.dumps([a[2] for a in anchors]),
            }
        )
    word_path = write_table(df, processed_dir / "word_samples.parquet")
    sentence_path = write_table(pd.DataFrame(sentence_rows), processed_dir / "sentence_samples.parquet")
    return {"word_samples": str(word_path), "sentence_samples": str(sentence_path), "eeg_dir": str(eeg_dir)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/zuco_mvp.yaml")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--output-report", default="outputs/reports/zuco_audit.json")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.audit_only:
        report = audit_raw_zuco(deep_get(cfg, "paths.raw_zuco_dir"))
        save_json(report, args.output_report)
        print(f"ZuCo audit saved to {args.output_report}")
        return

    if args.mock:
        result = create_mock_dataset(cfg, sentence_limit=args.sample_limit)
        print(json.dumps(result, indent=2))
        return

    if args.manifest:
        result = preprocess_manifest(cfg, args.manifest, sample_limit=args.sample_limit)
        print(json.dumps(result, indent=2))
        return

    report = audit_raw_zuco(deep_get(cfg, "paths.raw_zuco_dir"))
    save_json(report, args.output_report)
    raise SystemExit(
        "No real ZuCo parser was selected. Provide --manifest for manifest ingestion, "
        "or use --mock for the smoke pipeline. Raw audit was saved to "
        f"{args.output_report}."
    )


if __name__ == "__main__":
    main()
