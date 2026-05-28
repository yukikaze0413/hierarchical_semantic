from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hsb_eeg2text.config import deep_get, load_config
from hsb_eeg2text.preprocessing.zuco import LEXICON
from hsb_eeg2text.taxonomy.tree import Taxonomy
from hsb_eeg2text.utils.io import ensure_dir, read_table, save_json, write_table


FALLBACK_MIDS = {
    "person": "person",
    "place": "place",
    "object": "object",
    "action": "general_action",
    "attribute": "general_attribute",
    "event": "event",
    "abstract": "abstract",
}


def heuristic_path(keyword: str) -> tuple[str, str, str]:
    keyword = str(keyword).lower()
    if keyword in LEXICON:
        return LEXICON[keyword]
    if keyword.endswith("ing") or keyword.endswith("ed"):
        return ("action", "general_action", keyword.rstrip("ed"))
    if keyword.endswith("ly"):
        return ("attribute", "manner", keyword.rstrip("ly"))
    if keyword in {"doctor", "teacher", "student", "nurse", "scientist", "child", "patient"}:
        return ("person", FALLBACK_MIDS["person"], keyword)
    if keyword in {"hospital", "clinic", "library", "station", "room", "garden", "university"}:
        return ("place", FALLBACK_MIDS["place"], keyword)
    return ("object", "misc_object", keyword)


def build_taxonomy(config: dict, vocab_size: int | None = None, word_samples_path: str | Path | None = None, random_hierarchy: bool = False) -> dict:
    processed_dir = Path(deep_get(config, "paths.processed_zuco_dir"))
    taxonomy_dir = ensure_dir(deep_get(config, "paths.taxonomy_dir"))
    default_all = processed_dir / "word_samples_all.parquet"
    default_active = processed_dir / "word_samples.parquet"
    word_samples_path = Path(word_samples_path or (default_all if default_all.exists() else default_active))
    if not word_samples_path.exists() and word_samples_path.with_suffix(".csv").exists():
        word_samples_path = word_samples_path.with_suffix(".csv")
    samples = read_table(word_samples_path)
    vocab_size = int(vocab_size or deep_get(config, "data.vocabulary_size", 100))

    if "lemma" not in samples.columns:
        samples["lemma"] = samples["word"].astype(str).str.lower()
    counts = samples["lemma"].astype(str).str.lower().value_counts()
    keywords = counts.head(vocab_size).index.tolist()

    rows = []
    for keyword in keywords:
        subset = samples[samples["lemma"].astype(str).str.lower() == keyword]
        if {"coarse", "mid", "fine"}.issubset(subset.columns) and subset[["coarse", "mid", "fine"]].notna().all().all():
            coarse = str(subset["coarse"].iloc[0])
            mid = str(subset["mid"].iloc[0])
            fine = str(subset["fine"].iloc[0])
        else:
            coarse, mid, fine = heuristic_path(keyword)
        rows.append(
            {
                "keyword": keyword,
                "coarse": coarse,
                "mid": mid,
                "fine": fine,
                "frequency": int(counts.loc[keyword]),
            }
        )

    taxonomy_df = pd.DataFrame(rows).drop_duplicates("fine")
    if random_hierarchy:
        taxonomy_df = randomize_hierarchy(taxonomy_df, int(deep_get(config, "project.seed", 42)))
    edges = []
    for _, row in taxonomy_df.iterrows():
        edges.append({"parent": row["coarse"], "child": row["mid"], "edge_type": "coarse_to_mid"})
        edges.append({"parent": row["mid"], "child": row["fine"], "edge_type": "mid_to_fine"})
    edges_df = pd.DataFrame(edges).drop_duplicates()

    taxonomy_path = write_table(taxonomy_df, taxonomy_dir / "keyword_taxonomy.csv")
    edges_path = write_table(edges_df, taxonomy_dir / "hierarchy_edges.csv")
    annotation = annotate_processed_samples(config, taxonomy_df, word_samples_path)
    validation = Taxonomy(taxonomy_df).validate()
    save_json(validation, taxonomy_dir / "taxonomy_validation.json")
    return {
        "keyword_taxonomy_csv": str(taxonomy_path),
        "hierarchy_edges_csv": str(edges_path),
        "annotation": annotation,
        "validation": validation,
    }


def randomize_hierarchy(taxonomy_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = taxonomy_df.copy()
    coarse_values = out["coarse"].to_numpy().copy()
    mid_values = out["mid"].to_numpy().copy()
    rng.shuffle(coarse_values)
    rng.shuffle(mid_values)
    out["coarse"] = coarse_values
    out["mid"] = mid_values
    return out


def annotate_processed_samples(config: dict, taxonomy_df: pd.DataFrame, word_samples_path: str | Path) -> dict:
    processed_dir = Path(deep_get(config, "paths.processed_zuco_dir"))
    word_samples_path = Path(word_samples_path)
    samples = read_table(word_samples_path)
    mapping = {
        str(row["keyword"]).lower(): (str(row["coarse"]), str(row["mid"]), str(row["fine"]))
        for _, row in taxonomy_df.iterrows()
    }
    before = len(samples)
    if "lemma" not in samples.columns:
        samples["lemma"] = samples["word"].astype(str).str.lower()
    for col in ["coarse", "mid", "fine"]:
        if col not in samples.columns:
            samples[col] = None

    keep_rows = []
    for _, row in samples.iterrows():
        key = str(row["lemma"]).lower()
        if key not in mapping:
            continue
        coarse, mid, fine = mapping[key]
        row = row.copy()
        row["coarse"] = coarse
        row["mid"] = mid
        row["fine"] = fine
        keep_rows.append(row.to_dict())
    annotated = pd.DataFrame(keep_rows)
    annotated_path = write_table(annotated, processed_dir / "word_samples.parquet")

    sentence_rows = []
    if len(annotated):
        for sid, sdf in annotated.groupby("sentence_id", sort=False):
            anchors = sdf.sort_values("word_id")[["coarse", "mid", "fine"]].values.tolist()
            sentence_rows.append(
                {
                    "sentence_id": sid,
                    "sentence": sdf["sentence"].iloc[0],
                    "split": sdf["split"].iloc[0] if "split" in sdf.columns else "",
                    "anchors_json": json.dumps(anchors),
                    "fine_keywords_json": json.dumps([a[2] for a in anchors]),
                }
            )
    sentence_path = write_table(pd.DataFrame(sentence_rows), processed_dir / "sentence_samples.parquet")
    return {
        "word_samples": str(annotated_path),
        "sentence_samples": str(sentence_path),
        "rows_before": before,
        "rows_after": len(annotated),
        "rows_dropped_oov": before - len(annotated),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/zuco_mvp.yaml")
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--word-samples")
    parser.add_argument("--random-hierarchy", action="store_true")
    args = parser.parse_args()
    result = build_taxonomy(load_config(args.config), args.vocab_size, args.word_samples, args.random_hierarchy)
    print(result)


if __name__ == "__main__":
    main()
