# Experiment Plan

## Phase 0: Audit And Smoke Test

- Check server environment, CUDA, disk space, and Python dependencies.
- Audit `data/raw/zuco/` for file count, size, and likely MATLAB files.
- Run synthetic smoke test through preprocessing, taxonomy, one-epoch training, decoding, reconstruction, and evaluation.

## Phase 1: Hierarchical Decoding

- Dataset: ZuCo MVP.
- Vocabulary: top 100 content keywords.
- Taxonomy: heuristic/WordNet-ready coarse-mid-fine CSV with manual cleanup.
- Metrics: coarse/mid/fine Top-1, Top-5, MRR, hierarchical distance, LCA depth.

## Phase 2: Retrieval And Reconstruction

- Decode sentence-level hierarchical anchors.
- Retrieve examples with fine/mid/coarse overlap and semantic similarity.
- Reconstruct with `deepseek-v4-pro`.
- Compare flat keyword prompt, hierarchical anchors prompt, and oracle anchors prompt.

## Phase 3: Baselines And Ablations

- Flat keyword retrieval.
- Fine-only contrastive learning.
- Random hierarchy.
- Oracle hierarchy.
- No hierarchy loss.
- No curriculum.
- No hierarchy-aware RAG.
- Shuffled EEG-label control.

## Acceptance Criteria

- The numbered scripts complete a ZuCo MVP run in order.
- Each run saves config, logs, metrics, checkpoints, decoded anchors, and reconstructed sentences.
- API keys are read from environment variables only.
- Shuffled-label performance is near chance.
- Oracle hierarchy outperforms predicted hierarchy.
- Random hierarchy does not reproduce the gains of the real hierarchy.
