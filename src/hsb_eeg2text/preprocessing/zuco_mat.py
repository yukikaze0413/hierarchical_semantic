from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hsb_eeg2text.config import deep_get, load_config
from hsb_eeg2text.preprocessing.zuco import split_for_index
from hsb_eeg2text.utils.io import ensure_dir, save_json, write_table


@dataclass
class ParsedWord:
    subject_id: str
    sentence_id: str
    word_id: int
    word: str
    lemma: str
    sentence: str
    split: str
    eeg: np.ndarray


def extract_subject_id(path: str | Path) -> str:
    stem = Path(path).stem
    match = re.search(r"results([A-Za-z0-9]+?)(?:_[A-Za-z0-9]+)?$", stem)
    if match:
        return match.group(1)
    match = re.search(r"([A-Z]{2,}[A-Z0-9]*)", stem)
    return match.group(1) if match else stem


def normalize_field_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def find_feature_field(field_names: list[str], suffix: str) -> str | None:
    normalized = {name: normalize_field_name(name) for name in field_names}
    preferred = [
        f"mean_{suffix}_diff",
        f"{suffix}_diff",
        f"ffd_{suffix}_diff",
        f"trt_{suffix}_diff",
        f"gd_{suffix}_diff",
    ]
    for target in preferred:
        for original, norm in normalized.items():
            if norm == target or norm.endswith("_" + target):
                return original
    for original, norm in normalized.items():
        if suffix in norm.split("_") and norm.endswith("diff") and "mean" in norm:
            return original
    for original, norm in normalized.items():
        if norm.endswith(f"_{suffix}_diff"):
            return original
    return None


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "S", "O"}:
        flat = arr.flatten()
        if len(flat) == 1:
            return coerce_text(flat[0])
        return " ".join(coerce_text(x) for x in flat if coerce_text(x))
    if arr.dtype.kind in {"u", "i"} and arr.ndim > 0:
        try:
            chars = [chr(int(x)) for x in arr.flatten() if int(x) > 0]
            return "".join(chars)
        except Exception:
            return str(value)
    return str(value)


def coerce_vector(value: Any, expected_len: int = 48) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).squeeze()
    if arr.ndim == 0:
        return np.full(expected_len, np.nan, dtype=np.float32)
    arr = arr.flatten().astype(np.float32)
    if len(arr) == expected_len:
        return arr
    if len(arr) > expected_len:
        return arr[:expected_len]
    out = np.full(expected_len, np.nan, dtype=np.float32)
    out[: len(arr)] = arr
    return out


def stack_frequency_features(word_obj: dict[str, Any], suffixes: list[str], expected_pairs: int = 48) -> tuple[np.ndarray, dict[str, str | None]]:
    fields = list(word_obj.keys())
    vectors = []
    matched: dict[str, str | None] = {}
    for suffix in suffixes:
        field = find_feature_field(fields, suffix)
        matched[suffix] = field
        vector = coerce_vector(word_obj[field], expected_pairs) if field else np.full(expected_pairs, np.nan, dtype=np.float32)
        vectors.append(vector)
    stacked = np.stack(vectors, axis=1).astype(np.float32)
    return stacked[:, :, None], matched


def sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_") or "item"


def iter_mat_files(raw_dir: str | Path) -> list[Path]:
    return sorted(Path(raw_dir).rglob("*.mat"))


def source_scope_id(path: str | Path, raw_dir: str | Path) -> str:
    try:
        rel = Path(path).relative_to(Path(raw_dir))
    except ValueError:
        rel = Path(path).name
    return sanitize_id(str(rel).replace(".mat", ""))


def _todict_scipy(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.names:
            return [_todict_scipy(item) for item in obj.flatten()]
        if obj.size == 1:
            return _todict_scipy(obj.item())
        return [_todict_scipy(item) for item in obj.flatten()]
    if hasattr(obj, "_fieldnames"):
        return {field: _todict_scipy(getattr(obj, field)) for field in obj._fieldnames}
    return obj


def load_mat_scipy(path: Path) -> dict[str, Any]:
    from scipy.io import loadmat

    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    return {key: _todict_scipy(value) for key, value in data.items() if not key.startswith("__")}


def _read_h5_value(h5, obj):
    import h5py

    if isinstance(obj, h5py.Reference):
        if not obj:
            return None
        return _read_h5_value(h5, h5[obj])
    if isinstance(obj, h5py.Dataset):
        data = obj[()]
        if isinstance(data, np.ndarray) and data.dtype == h5py.ref_dtype:
            return [_read_h5_value(h5, ref) for ref in data.flatten()]
        if obj.dtype.kind in {"u", "i"} and obj.ndim >= 1 and max(obj.shape, default=0) > 1:
            try:
                return "".join(chr(int(x)) for x in np.asarray(data).flatten() if int(x) > 0)
            except Exception:
                return data
        return data
    if isinstance(obj, h5py.Group):
        return {key: _read_h5_value(h5, obj[key]) for key in obj.keys()}
    return obj


def load_mat_h5(path: Path) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as h5:
        return {key: _read_h5_value(h5, h5[key]) for key in h5.keys()}


def load_mat_any(path: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(path)
    h5py_error = None
    try:
        return load_mat_h5(path), "h5py"
    except Exception as exc:
        h5py_error = str(exc)
    try:
        mat = load_mat_scipy(path)
        mat["_loader_errors"] = {"h5py_error": h5py_error, "scipy_error": None}
        return mat, "scipy"
    except Exception as exc:
        raise RuntimeError(json.dumps({"h5py_error": h5py_error, "scipy_error": str(exc)}))


def as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.flatten().tolist()
    return [value]


def get_first(obj: dict[str, Any], names: list[str], default: Any = None) -> Any:
    normalized = {normalize_field_name(key): key for key in obj.keys()}
    for name in names:
        key = normalized.get(normalize_field_name(name))
        if key is not None:
            return obj[key]
    return default


def sample_word_fields(sentences: list[Any], max_fields: int = 80) -> list[str]:
    for sentence_obj in sentences:
        if not isinstance(sentence_obj, dict):
            continue
        word_objs = [word for word in as_list(get_first(sentence_obj, ["word"])) if isinstance(word, dict)]
        if word_objs:
            return sorted(list(word_objs[0].keys()))[:max_fields]
    return []


def sentence_text_from(sentence_obj: dict[str, Any], words: list[dict[str, Any]]) -> str:
    text = get_first(sentence_obj, ["content", "sentence", "text", "raw_text"], "")
    text = coerce_text(text).strip()
    if text and text != "None":
        return text
    word_texts = [coerce_text(get_first(word, ["content", "word", "text"], "")).strip() for word in words]
    return " ".join(w for w in word_texts if w)


def parse_sentence_data(
    mat: dict[str, Any],
    subject_id: str,
    suffixes: list[str],
    train_ratio: float,
    val_ratio: float,
    source_file: str,
) -> tuple[list[ParsedWord], dict[str, Any]]:
    sentence_data = get_first(mat, ["sentenceData"])
    sentences = as_list(sentence_data)
    report = {
        "source_file": source_file,
        "subject_id": subject_id,
        "has_sentenceData": sentence_data is not None,
        "sentenceData_type": type(sentence_data).__name__,
        "sentence_count": len(sentences),
        "word_count": 0,
        "top_level_keys": sorted(list(mat.keys()))[:80],
        "word_field_sample": sample_word_fields(sentences),
        "used_feature_fields": {},
        "missing_suffix_counts": {suffix: 0 for suffix in suffixes},
        "skipped_missing_word": 0,
        "skipped_missing_eeg": 0,
    }
    rows: list[ParsedWord] = []
    global_sentence_count = max(len(sentences), 1)
    for sent_idx, sentence_obj in enumerate(sentences):
        if not isinstance(sentence_obj, dict):
            continue
        words_value = get_first(sentence_obj, ["word"])
        word_objs = [word for word in as_list(words_value) if isinstance(word, dict)]
        if not word_objs:
            report["skipped_missing_word"] += 1
            continue
        sentence = sentence_text_from(sentence_obj, word_objs)
        split = split_for_index(sent_idx, global_sentence_count, train_ratio, val_ratio)
        sentence_id = f"{sanitize_id(subject_id)}_sent_{sent_idx:05d}"
        for word_idx, word_obj in enumerate(word_objs):
            word_text = coerce_text(get_first(word_obj, ["content", "word", "text"], "")).strip()
            if not word_text or word_text == "None":
                word_text = f"word_{word_idx}"
            lemma = coerce_text(get_first(word_obj, ["lemma"], word_text)).strip().lower() or word_text.lower()
            eeg, matched = stack_frequency_features(word_obj, suffixes)
            for suffix, field in matched.items():
                if field:
                    report["used_feature_fields"].setdefault(suffix, field)
                else:
                    report["missing_suffix_counts"][suffix] += 1
            if np.isnan(eeg).all():
                report["skipped_missing_eeg"] += 1
                continue
            rows.append(
                ParsedWord(
                    subject_id=subject_id,
                    sentence_id=sentence_id,
                    word_id=word_idx,
                    word=word_text,
                    lemma=lemma,
                    sentence=sentence,
                    split=split,
                    eeg=eeg,
                )
            )
            report["word_count"] += 1
    return rows, report


def apply_train_zscore(parsed: list[ParsedWord]) -> tuple[list[ParsedWord], dict[str, Any]]:
    if not parsed:
        return parsed, {"applied": False}
    train_arrays = [item.eeg for item in parsed if item.split == "train"]
    if not train_arrays:
        train_arrays = [item.eeg for item in parsed]
    stack = np.stack(train_arrays, axis=0)
    mean = np.nanmean(stack, axis=0)
    mean = np.where(np.isnan(mean), 0.0, mean)
    filled_train = np.where(np.isnan(stack), mean[None, ...], stack)
    std = np.std(filled_train, axis=0)
    std = np.where((std < 1e-6) | np.isnan(std), 1.0, std)
    out = []
    for item in parsed:
        eeg = np.where(np.isnan(item.eeg), mean, item.eeg)
        eeg = ((eeg - mean) / std).astype(np.float32)
        out.append(ParsedWord(**{**item.__dict__, "eeg": eeg}))
    return out, {
        "applied": True,
        "mean_shape": list(mean.shape),
        "std_shape": list(std.shape),
        "train_items": len(train_arrays),
    }


def write_parsed_outputs(config: dict[str, Any], parsed: list[ParsedWord], report: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    processed_dir = ensure_dir(deep_get(config, "paths.processed_zuco_dir"))
    eeg_dir = ensure_dir(deep_get(config, "paths.eeg_word_dir"))
    word_rows = []
    sentence_map: dict[str, dict[str, Any]] = {}
    for item in parsed:
        sample_id = sanitize_id(f"{item.subject_id}_{item.sentence_id}_w{item.word_id:04d}")
        eeg_path = eeg_dir / f"{sample_id}.npy"
        np.save(eeg_path, item.eeg.astype(np.float32))
        word_rows.append(
            {
                "sample_id": sample_id,
                "subject_id": item.subject_id,
                "sentence_id": item.sentence_id,
                "word_id": item.word_id,
                "word": item.word,
                "lemma": item.lemma,
                "sentence": item.sentence,
                "eeg_path": str(eeg_path),
                "split": item.split,
            }
        )
        sentence_entry = sentence_map.setdefault(
            item.sentence_id,
            {"sentence_id": item.sentence_id, "sentence": item.sentence, "split": item.split, "words": []},
        )
        sentence_entry["words"].append(item.lemma)

    sentence_rows = [
        {
            "sentence_id": entry["sentence_id"],
            "sentence": entry["sentence"],
            "split": entry["split"],
            "anchors_json": "[]",
            "fine_keywords_json": json.dumps(entry["words"]),
        }
        for entry in sentence_map.values()
    ]
    word_path = write_table(pd.DataFrame(word_rows), processed_dir / "word_samples_all.parquet")
    sentence_path = write_table(pd.DataFrame(sentence_rows), processed_dir / "sentence_samples_all.parquet")
    # Active files start as all files; taxonomy construction will filter/annotate them.
    active_word_path = write_table(pd.DataFrame(word_rows), processed_dir / "word_samples.parquet")
    active_sentence_path = write_table(pd.DataFrame(sentence_rows), processed_dir / "sentence_samples.parquet")
    save_json(report, Path(deep_get(config, "paths.reports_dir")) / "zuco_mat_structure.json")
    return {
        "word_samples_all": str(word_path),
        "sentence_samples_all": str(sentence_path),
        "word_samples": str(active_word_path),
        "sentence_samples": str(active_sentence_path),
        "rows": len(word_rows),
        "sentences": len(sentence_rows),
        "structure_report": str(Path(deep_get(config, "paths.reports_dir")) / "zuco_mat_structure.json"),
    }


def preprocess_zuco_mat(config: dict[str, Any], sample_limit: int | None = None) -> dict[str, Any]:
    from tqdm.auto import tqdm

    raw_dir = Path(deep_get(config, "paths.raw_zuco_dir"))
    suffixes = list(deep_get(config, "preprocessing.zuco_frequency_suffixes", ["t1", "t2", "a1", "a2", "b1", "b2", "g1", "g2"]))
    train_ratio = float(deep_get(config, "data.train_ratio", 0.8))
    val_ratio = float(deep_get(config, "data.val_ratio", 0.1))
    parsed: list[ParsedWord] = []
    file_reports = []
    mat_files = iter_mat_files(raw_dir)
    for mat_path in tqdm(mat_files, desc="Parsing ZuCo .mat files", unit="file"):
        subject_id = sanitize_id(f"{source_scope_id(mat_path, raw_dir)}_{extract_subject_id(mat_path)}")
        try:
            mat, loader = load_mat_any(mat_path)
            rows, report = parse_sentence_data(mat, subject_id, suffixes, train_ratio, val_ratio, str(mat_path))
            report["loader"] = loader
            if "_loader_errors" in mat:
                report["loader_errors"] = mat["_loader_errors"]
            parsed.extend(rows)
        except Exception as exc:
            loader_errors = None
            try:
                loader_errors = json.loads(str(exc))
            except Exception:
                pass
            report = {
                "source_file": str(mat_path),
                "subject_id": subject_id,
                "error": str(exc),
                "loader_errors": loader_errors,
                "has_sentenceData": False,
                "sentence_count": 0,
                "word_count": 0,
            }
        file_reports.append(report)
        if sample_limit and len(parsed) >= sample_limit:
            parsed = parsed[:sample_limit]
            break
    print(f"Parsed word-level EEG samples: {len(parsed)}", flush=True)
    parsed, zscore_report = apply_train_zscore(parsed)
    report = {
        "raw_dir": str(raw_dir),
        "mat_file_count": len(mat_files),
        "parsed_word_count": len(parsed),
        "eeg_shape": list(parsed[0].eeg.shape) if parsed else None,
        "zscore": zscore_report,
        "files": file_reports,
    }
    if not parsed:
        report_path = Path(deep_get(config, "paths.reports_dir")) / "zuco_mat_structure.json"
        save_json(report, report_path)
        raise RuntimeError(f"No usable ZuCo word-level EEG features were parsed. Inspect {report_path}.")
    return write_parsed_outputs(config, parsed, report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/zuco_mvp.yaml")
    parser.add_argument("--sample-limit", type=int)
    args = parser.parse_args()
    result = preprocess_zuco_mat(load_config(args.config), sample_limit=args.sample_limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
