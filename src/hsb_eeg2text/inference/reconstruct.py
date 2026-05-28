from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from hsb_eeg2text.config import deep_get, load_config
from hsb_eeg2text.inference.llm_backends import build_llm_backend
from hsb_eeg2text.inference.rag import load_sentence_index, retrieve_examples
from hsb_eeg2text.models.text_encoder import HashTextEncoder, SentenceTransformerTextEncoder
from hsb_eeg2text.utils.io import read_jsonl, write_jsonl


def aggregate_sentence_anchors(decoded_rows: list[dict], threshold: float, max_anchors: int = 8) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    for row in decoded_rows:
        sid = row["sentence_id"]
        pred = row["predictions"][0] if row.get("predictions") else {}
        pred = dict(pred)
        if pred.get("fine_score", 0.0) < threshold:
            pred["fine"] = None
        grouped.setdefault(sid, {"sentence_id": sid, "gold_sentence": row["sentence"], "anchors": []})
        grouped[sid]["anchors"].append(pred)
    for item in grouped.values():
        dedup = {}
        for anchor in item["anchors"]:
            key = (anchor.get("coarse"), anchor.get("mid"), anchor.get("fine"))
            dedup[key] = max(dedup.get(key, anchor), anchor, key=lambda a: a.get("score", 0.0))
        item["anchors"] = sorted(dedup.values(), key=lambda a: a.get("score", 0.0), reverse=True)[:max_anchors]
    return grouped


def build_prompt(anchors: list[dict], examples: list[dict], mode: str = "hierarchical_anchors") -> str:
    anchor_lines = []
    for idx, anchor in enumerate(anchors, 1):
        fine = anchor.get("fine") or "[fine uncertain]"
        if mode == "flat_keywords":
            anchor_lines.append(f"{idx}. {fine}, confidence: {anchor.get('fine_score', anchor.get('score', 0.0)):.3f}")
        else:
            anchor_lines.append(
                f"{idx}. {anchor.get('coarse')} > {anchor.get('mid')} > {fine}, confidence: {anchor.get('fine_score', anchor.get('score', 0.0)):.3f}"
            )
    example_lines = [f"{idx}. {item['sentence']}" for idx, item in enumerate(examples, 1)]
    return f"""You are reconstructing one fluent English sentence from EEG-derived hierarchical semantic anchors.

The EEG decoder is more reliable at coarse and mid semantic levels than exact word level.
Use fine keywords when confidence is high. When fine keywords are uncertain, preserve the mid-level meaning.

Return compact JSON with keys: entities, action, sentence.

Semantic anchors:
{chr(10).join(anchor_lines)}

Retrieved reference sentences:
{chr(10).join(example_lines)}
"""


def strip_code_fence(text: str) -> str | None:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def extract_json_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def parse_generation(text: str) -> dict:
    candidates = [text.strip()]
    fenced = strip_code_fence(text)
    if fenced:
        candidates.append(fenced)
    extracted = extract_json_object(text)
    if extracted:
        candidates.append(extracted)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            parsed.setdefault("entities", [])
            parsed.setdefault("action", "")
            parsed.setdefault("sentence", "")
            parsed["raw_response_text"] = text
            return parsed
        except Exception:
            continue
    cleaned = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()
    return {"entities": [], "action": "", "sentence": cleaned, "raw_response_text": text}


def build_rag_encoder(config: dict, backend_name: str | None = None):
    weights = deep_get(config, "inference.rag_weights", {})
    if weights.get("semantic", 0.0) <= 0:
        return None
    if backend_name == "mock":
        return HashTextEncoder(int(deep_get(config, "model.embedding_dim", 256)))
    rag_cfg = deep_get(config, "model.rag_encoder", {})
    if rag_cfg.get("backend") == "sentence_transformer":
        return SentenceTransformerTextEncoder(rag_cfg["model_name"], int(deep_get(config, "model.embedding_dim", 256)))
    if rag_cfg.get("backend") == "hash":
        return HashTextEncoder(int(deep_get(config, "model.embedding_dim", 256)))
    return None


def oracle_anchors_from_decoded(rows: list[dict]) -> list[dict]:
    anchors = []
    for row in rows:
        gold = row.get("gold", {})
        anchors.append(
            {
                "coarse": gold.get("coarse"),
                "mid": gold.get("mid"),
                "fine": gold.get("fine"),
                "score": 1.0,
                "fine_score": 1.0,
            }
        )
    return anchors


def reconstruct(
    config: dict,
    decoded_path: str | Path,
    backend_name: str | None = None,
    output_path: str | Path | None = None,
    reconstruction_mode: str = "hierarchical_anchors",
) -> dict:
    processed_dir = Path(deep_get(config, "paths.processed_zuco_dir"))
    sentence_path = processed_dir / "sentence_samples.parquet"
    if not sentence_path.exists():
        sentence_path = processed_dir / "sentence_samples.csv"
    decoded_rows = read_jsonl(decoded_path)
    groups = aggregate_sentence_anchors(
        decoded_rows,
        threshold=float(deep_get(config, "inference.fine_confidence_threshold", 0.3)),
    )
    if reconstruction_mode == "oracle_anchors":
        by_sentence: dict[str, list[dict]] = {}
        for row in decoded_rows:
            by_sentence.setdefault(row["sentence_id"], []).append(row)
        for sid, rows_for_sentence in by_sentence.items():
            if sid in groups:
                groups[sid]["anchors"] = oracle_anchors_from_decoded(rows_for_sentence)
    rag_encoder = build_rag_encoder(config, backend_name)
    index = load_sentence_index(sentence_path, text_encoder=rag_encoder)
    backend = build_llm_backend(config, backend_name)
    rows = []
    for sid, item in groups.items():
        examples = []
        if reconstruction_mode != "no_rag":
            examples = retrieve_examples(
                item["anchors"],
                index,
                top_k=int(deep_get(config, "inference.rag_top_k", 5)),
                weights=deep_get(config, "inference.rag_weights"),
                query_sentence_id=sid,
                allowed_splits={"train"},
                text_encoder=rag_encoder,
            )
        prompt = build_prompt(item["anchors"], examples, mode=reconstruction_mode)
        response = backend.generate(prompt)
        parsed = parse_generation(response.text)
        rows.append(
            {
                "sentence_id": sid,
                "gold_sentence": item["gold_sentence"],
                "anchors": item["anchors"],
                "retrieved_examples": examples,
                "generated_sentence": parsed.get("sentence", response.text),
                "parsed": parsed,
                "backend": backend_name or deep_get(config, "llm.default_backend"),
                "reconstruction_mode": reconstruction_mode,
                "raw_response_text": parsed.get("raw_response_text", response.text),
                "raw_response": response.raw,
            }
        )
    if output_path is None:
        output_path = Path(deep_get(config, "paths.reconstructed_dir")) / "reconstructed.jsonl"
    write_jsonl(rows, output_path)
    return {"reconstructed_path": str(output_path), "rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/zuco_mvp.yaml")
    parser.add_argument("--decoded", required=True)
    parser.add_argument("--backend")
    parser.add_argument("--output")
    parser.add_argument("--reconstruction-mode", default="hierarchical_anchors", choices=["hierarchical_anchors", "flat_keywords", "oracle_anchors", "no_rag"])
    args = parser.parse_args()
    print(json.dumps(reconstruct(load_config(args.config), args.decoded, args.backend, args.output, args.reconstruction_mode), indent=2))


if __name__ == "__main__":
    main()
