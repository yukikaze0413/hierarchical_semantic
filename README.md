# Hierarchical Semantic Bottleneck EEG-to-Text

This repository implements the ZuCo MVP for the Hierarchical Semantic Bottleneck EEG-to-Text idea:

```text
EEG -> coarse semantic category -> mid-level concept -> fine keyword -> RAG + LLM sentence reconstruction
```

The first milestone is a reproducible experimental loop, not an SOTA model. It is designed for a remote Linux server with a 4090-class GPU.

## Quick Start On The Server

```bash
conda env create -f environment.yml
conda activate hsb-eeg2text
pip install -e .
chmod +x scripts/*.sh

scripts/00_check_env.sh
scripts/02_check_data.sh
scripts/09_run_smoke_test.sh
```

For DeepSeek reconstruction:

```bash
export DEEPSEEK_API_KEY="..."
scripts/07_reconstruct_sentences.sh
```

## Pipeline

```text
00_check_env.sh
01_prepare_env.sh
02_check_data.sh
03_preprocess_zuco.sh
04_build_taxonomy.sh
05_train_model.sh
06_decode_anchors.sh
07_reconstruct_sentences.sh
08_evaluate.sh
09_run_smoke_test.sh
```

Each bash script is interactive, prints the active config and output log path, and asks before running. Long-running stages use `tmux` when available.

## Default MVP Choices

- Dataset: ZuCo
- Vocabulary: 100 keywords
- EEG feature shape: `[48, 5, 10]`
- EEG encoder: Deep4-style CNN
- Text encoder: `sentence-transformers/all-MiniLM-L6-v2`
- RAG encoder: `sentence-transformers/all-MiniLM-L6-v2`
- LLM backend: DeepSeek OpenAI-compatible API
- Formal LLM model: `deepseek-v4-pro`
- Debug LLM model: `deepseek-v4-flash`

## Data Layout

```text
data/raw/zuco/                  # Put licensed ZuCo files here
data/raw/zuco/task1-SR/Matlab_files/*.mat
data/raw/zuco/task2-NR/Matlab_files/*.mat
data/processed/zuco/word_samples.parquet
data/processed/zuco/sentence_samples.parquet
data/processed/zuco/word_samples_all.parquet
data/processed/zuco/sentence_samples_all.parquet
data/processed/zuco/eeg_word/{sample_id}.npy
data/taxonomy/keyword_taxonomy.csv
data/taxonomy/hierarchy_edges.csv
```

The smoke test creates a synthetic ZuCo-like dataset and does not require real ZuCo files.

## Storage Guidance

- API-only LLM: at least 150 GB.
- Recommended MVP: at least 200 GB.
- Many ablations/checkpoints: around 500 GB.
- Long-term raw + processed + local LLM caches: around 1 TB.

## Real ZuCo `.mat` Preprocessing

The first real-data mode reads OSF ZuCo Matlab feature files. It expects MATLAB files containing `sentenceData`, with word-level frequency EEG features under `sentenceData.word`.

Use:

```bash
scripts/03_preprocess_zuco.sh
# choose: ZuCo .mat frequency features
```

The parser prioritizes 48 electrode-pair diff frequency features with suffixes:

```text
t1,t2,a1,a2,b1,b2,g1,g2
```

It writes unfiltered files:

```text
word_samples_all.parquet
sentence_samples_all.parquet
```

Then taxonomy construction creates active top-k training files:

```text
word_samples.parquet
sentence_samples.parquet
```

This mode uses OSF word-level frequency features, not raw continuous EEG. `rawEEG` parsing is a later extension.

## Manifest Preprocessing Note

If the `.mat` files do not match a supported structure, generate a manifest with columns:

```text
subject_id,sentence_id,word_id,word,lemma,sentence,eeg_path
```

Manifest mode expects `eeg_path` to point to either already cleaned word-level epochs `[channels,time]` or precomputed features matching `data.eeg_shape`. It does not perform raw continuous EEG bandpass/notch/ICA processing.

After manifest preprocessing, run taxonomy construction. That step annotates `word_samples` with `coarse/mid/fine`, filters words outside the configured vocabulary, and rebuilds sentence-level anchors before training.
