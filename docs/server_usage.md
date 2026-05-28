# Server Usage

## Expected Server

- Linux
- NVIDIA 4090-class GPU
- CUDA-compatible driver
- Conda
- Optional but recommended: `tmux`
- Storage: 200 GB minimum, 500 GB comfortable

## Environment

```bash
conda env create -f environment.yml
conda activate hsb-eeg2text
pip install -e .
```

If PyTorch CUDA resolution fails, install PyTorch from the official selector for the server's CUDA version, then rerun:

```bash
pip install -e .
```

## API Keys

Do not write API keys into config files. For DeepSeek:

```bash
export DEEPSEEK_API_KEY="..."
```

## Logs

Every interactive stage writes a timestamped log under:

```text
outputs/logs/
```

Training and reconstruction scripts use `tmux` for long-running jobs if it is installed.

## Manifest Mode

Manifest ingestion is for already-cleaned word-level epochs or precomputed features matching `data.eeg_shape`. The taxonomy stage annotates the resulting `word_samples` with `coarse/mid/fine` and filters out words outside the configured MVP vocabulary.

## ZuCo `.mat` Mode

Place OSF Matlab feature files under paths such as:

```text
data/raw/zuco/task1-SR/Matlab_files/*.mat
data/raw/zuco/task2-NR/Matlab_files/*.mat
```

Then run:

```bash
scripts/03_preprocess_zuco.sh
```

and select `ZuCo .mat frequency features`. The parser uses `sentenceData -> word` frequency diff features and saves `[48,8,1]` EEG tensors. It also writes `outputs/reports/zuco_mat_structure.json` so unsupported field layouts can be inspected.
