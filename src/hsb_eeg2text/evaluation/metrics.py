from __future__ import annotations

import re

from hsb_eeg2text.taxonomy.tree import hierarchical_distance, lca_depth


def topk_hit(predictions: list[dict], gold: dict, k: int, level: str) -> bool:
    return any(pred.get(level) == gold.get(level) for pred in predictions[:k])


def decode_metrics(decoded_rows: list[dict], top_ks: tuple[int, ...] = (1, 5)) -> dict:
    metrics: dict[str, float] = {}
    n = max(len(decoded_rows), 1)
    for level in ["coarse", "mid", "fine"]:
        for k in top_ks:
            metrics[f"{level}_top{k}"] = sum(topk_hit(row.get("predictions", []), row["gold"], k, level) for row in decoded_rows) / n
    distances = []
    depths = []
    for row in decoded_rows:
        if not row.get("predictions"):
            continue
        pred = row["predictions"][0]
        gold_path = [row["gold"]["coarse"], row["gold"]["mid"], row["gold"]["fine"]]
        pred_path = [pred.get("coarse"), pred.get("mid"), pred.get("fine")]
        distances.append(hierarchical_distance(gold_path, pred_path))
        depths.append(lca_depth(gold_path, pred_path))
    metrics["hierarchical_distance"] = sum(distances) / max(len(distances), 1)
    metrics["lca_depth"] = sum(depths) / max(len(depths), 1)
    return metrics


def reconstruction_metrics(rows: list[dict]) -> dict:
    n = max(len(rows), 1)
    keyword_coverage = []
    concept_coverage = []
    hierarchy_coverage = []
    token_f1_scores = []
    rouge_l_scores = []
    retrieval_hits = {5: [], 10: []}
    for row in rows:
        generated = row.get("generated_sentence", "").lower()
        gold = row.get("gold_sentence", "").lower()
        anchors = row.get("anchors", [])
        fine_terms = [a.get("fine") for a in anchors if a.get("fine")]
        mid_terms = [a.get("mid") for a in anchors if a.get("mid")]
        coarse_terms = [a.get("coarse") for a in anchors if a.get("coarse")]
        if not fine_terms:
            keyword_coverage.append(0.0)
        else:
            keyword_coverage.append(sum(term.lower() in generated for term in fine_terms) / len(fine_terms))
        concept_terms = mid_terms + coarse_terms
        concept_coverage.append(sum(str(term).lower().replace("_", " ") in generated for term in concept_terms) / max(len(concept_terms), 1))
        hierarchy_terms = fine_terms + mid_terms + coarse_terms
        hierarchy_coverage.append(sum(str(term).lower().replace("_", " ") in generated for term in hierarchy_terms) / max(len(hierarchy_terms), 1))
        token_f1_scores.append(token_f1(gold, generated))
        rouge_l_scores.append(rouge_l(gold, generated))
        retrieved = row.get("retrieved_examples", [])
        for k in retrieval_hits:
            retrieval_hits[k].append(any(str(ex.get("sentence", "")).lower() == gold for ex in retrieved[:k]))
    optional, skipped = optional_generation_metrics(rows)
    return {
        "reconstructed_sentences": len(rows),
        "keyword_coverage": sum(keyword_coverage) / n,
        "concept_coverage": sum(concept_coverage) / n,
        "hierarchy_coverage": sum(hierarchy_coverage) / n,
        "token_f1": sum(token_f1_scores) / n,
        "rouge_l": sum(rouge_l_scores) / n,
        "retrieval_top5": sum(retrieval_hits[5]) / n,
        "retrieval_top10": sum(retrieval_hits[10]) / n,
        "skipped_metrics": skipped,
        **optional,
    }


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(reference: str, prediction: str) -> float:
    ref = tokenize(reference)
    pred = tokenize(prediction)
    if not ref or not pred:
        return 0.0
    ref_counts = {}
    for tok in ref:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1
    overlap = 0
    for tok in pred:
        if ref_counts.get(tok, 0) > 0:
            overlap += 1
            ref_counts[tok] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def lcs_len(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            old = dp[j]
            if x == y:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = old
    return dp[-1]


def rouge_l(reference: str, prediction: str) -> float:
    ref = tokenize(reference)
    pred = tokenize(prediction)
    if not ref or not pred:
        return 0.0
    lcs = lcs_len(ref, pred)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def optional_generation_metrics(rows: list[dict]) -> tuple[dict, list[str]]:
    metrics = {}
    skipped = []
    refs = [row.get("gold_sentence", "") for row in rows]
    preds = [row.get("generated_sentence", "") for row in rows]
    if rows:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            ref_emb = model.encode(refs, normalize_embeddings=True, show_progress_bar=False)
            pred_emb = model.encode(preds, normalize_embeddings=True, show_progress_bar=False)
            metrics["sbert_similarity"] = float(np.mean(np.sum(ref_emb * pred_emb, axis=1)))
        except Exception as exc:
            skipped.append(f"sbert_similarity: {exc}")
        try:
            from bert_score import score

            _, _, f1 = score(preds, refs, lang="en", verbose=False)
            metrics["bertscore_f1"] = float(f1.mean().item())
        except Exception as exc:
            skipped.append(f"bertscore_f1: {exc}")
    return metrics, skipped
