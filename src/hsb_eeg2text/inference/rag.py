from __future__ import annotations

import json
from pathlib import Path

from hsb_eeg2text.utils.io import read_table


def load_sentence_index(sentence_samples_path: str | Path, text_encoder=None) -> list[dict]:
    df = read_table(sentence_samples_path)
    index = []
    for _, row in df.iterrows():
        anchors = json.loads(row.get("anchors_json", "[]"))
        coarse = {a[0] for a in anchors if len(a) == 3}
        mid = {a[1] for a in anchors if len(a) == 3}
        fine = {a[2] for a in anchors if len(a) == 3}
        item = {
            "sentence_id": str(row["sentence_id"]),
            "sentence": str(row["sentence"]),
            "split": str(row.get("split", "")),
            "anchors": anchors,
            "coarse": sorted(coarse),
            "mid": sorted(mid),
            "fine": sorted(fine),
        }
        if text_encoder is not None:
            item["embedding"] = text_encoder.encode([item["sentence"]])[0].astype(float).tolist()
        index.append(item)
    return index


def anchor_query_text(anchor_paths: list[dict]) -> str:
    parts = []
    for path in anchor_paths:
        fine = path.get("fine") or "uncertain"
        parts.append(f"{path.get('coarse')} {path.get('mid')} {fine}")
    return " ; ".join(parts)


def cosine(a, b) -> float:
    import numpy as np

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def retrieve_examples(
    anchor_paths: list[dict],
    index: list[dict],
    top_k: int = 5,
    weights: dict | None = None,
    query_sentence_id: str | None = None,
    allowed_splits: set[str] | None = None,
    text_encoder=None,
) -> list[dict]:
    weights = weights or {"fine": 0.4, "mid": 0.3, "coarse": 0.1, "semantic": 0.2}
    allowed_splits = allowed_splits or {"train"}
    query = {
        "coarse": {p["coarse"] for p in anchor_paths if p.get("coarse")},
        "mid": {p["mid"] for p in anchor_paths if p.get("mid")},
        "fine": {p["fine"] for p in anchor_paths if p.get("fine")},
    }
    query_embedding = None
    if text_encoder is not None and weights.get("semantic", 0.0) > 0:
        query_embedding = text_encoder.encode([anchor_query_text(anchor_paths)])[0]

    scored = []
    for item in index:
        if query_sentence_id and item["sentence_id"] == query_sentence_id:
            continue
        if allowed_splits and item.get("split") not in allowed_splits:
            continue
        item_coarse = set(item["coarse"])
        item_mid = set(item["mid"])
        item_fine = set(item["fine"])
        semantic_sim = 0.0
        if query_embedding is not None and "embedding" in item:
            import numpy as np

            semantic_sim = (cosine(np.asarray(query_embedding), np.asarray(item["embedding"])) + 1.0) / 2.0
        score = (
            weights.get("fine", 0.4) * jaccard(query["fine"], item_fine)
            + weights.get("mid", 0.3) * jaccard(query["mid"], item_mid)
            + weights.get("coarse", 0.1) * jaccard(query["coarse"], item_coarse)
            + weights.get("semantic", 0.0) * semantic_sim
        )
        clean_item = {key: value for key, value in item.items() if key != "embedding"}
        scored.append((score, clean_item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{**item, "score": score} for score, item in scored[:top_k]]
