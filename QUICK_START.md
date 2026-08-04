# Quick Start Guide

For experienced users who want to get training running quickly.

## 1. Setup (2 minutes)

```bash
# Clone and setup
git clone <repo> && cd SLM-from-scratch
python3 -m venv venv && source venv/bin/activate
pip install torch sentencepiece datasets pyyaml tqdm numpy --index-url https://download.pytorch.org/whl/cu118

# Verify setup
python3 scripts/validate_setup.py
```

## 2. Prepare Data (5-10 minutes)

```bash
# Download and preprocess FineWeb-Edu (100k docs, ~500M tokens)
python3 scripts/preprocess_fineweb.py \
  --limit 100000 \
  --vocab-size 50000 \
  --context-length 2048
```

## 3. Train (Start here, runs indefinitely)

```bash
# Small model (~90M params) - RECOMMENDED
python3 scripts/train_real.py \
  --data-dir datasets/fineweb_edu/processed/train \
  --model-size small \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --max-steps 100000 \
  --checkpoint-dir checkpoints

# In another terminal, monitor:
tail -f checkpoints/logs/history.jsonl
nvidia-smi -l 1
```

## 4. Resume Training

```bash
# Automatically resumes from latest checkpoint
python3 scripts/train_real.py \
  --data-dir datasets/fineweb_edu/processed/train \
  --model-size small \
  --checkpoint-dir checkpoints
```

## Key Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--model-size` | `small` | `small` (~90M) or `medium` (~251M) |
| `--batch-size` | `4` | Reduce to 2 if OOM |
| `--context-length` | `2048` | Reduce to 1024 if OOM |
| `--gradient-accumulation-steps` | `4` | Effective batch = batch_size × grad_accum |
| `--gradient-checkpointing` | False | Enable to save memory (~40% reduction) |
| `--max-steps` | `100000` | Total training steps |
| `--learning-rate` | `5e-4` | Adjust if loss not decreasing |

## Expected Performance

- **Throughput:** 800-1200 tokens/sec
- **GPU Memory:** 11-12GB
- **Loss (step 1):** ~5.5-6.0
- **Loss (step 1000):** ~3.5-4.5 (30-40% decrease)
- **Time for 10k steps:** ~3-5 hours

## Troubleshooting Quick Fixes

| Error | Fix |
|-------|-----|
| `CUDA out of memory` | Add `--gradient-checkpointing` or reduce `--batch-size` |
| `Loss not decreasing` | Try `--learning-rate 1e-3` |
| `GPU util 0%` | Check disk I/O, shards may be on slow disk |
| `Checkpoint corrupted` | Delete `.pt` file, training resumes from previous |

See [TRAINING_MONITORING.md](TRAINING_MONITORING.md) and [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for details.

## File Structure After Setup

```
SLM-from-scratch/
├── checkpoints/
│   ├── model_step_500.pt
│   ├── model_step_1000.pt
│   └── logs/
│       └── history.jsonl
├── datasets/
│   └── fineweb_edu/processed/train/
│       ├── metadata.json
│       ├── shard_000.bin
│       └── ...
└── slm/
```
