# SLM Training Setup Guide for RTX 3060

This guide walks you through setting up and running SLM training on an RTX 3060 12GB GPU with 64GB RAM.

## System Requirements

- **GPU**: NVIDIA RTX 3060 (12GB VRAM)
- **RAM**: 64GB
- **Storage**: Minimum 500GB for dataset shards
- **OS**: Linux (Ubuntu 20.04+) or similar

## Step 1: Environment Setup

### 1.1 Clone Repository

```bash
cd /path/to/your/training/machine
git clone <repository-url> SLM-from-scratch
cd SLM-from-scratch
```

### 1.2 Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch (with CUDA 11.8 support for RTX 3060)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install sentencepiece datasets pyyaml tqdm numpy

# Verify PyTorch installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step 2: Prepare Dataset

The dataset is preprocessed into binary shards for fast memmapped training.

### 2.1 Download and Preprocess FineWeb-Edu

```bash
# For RTX 3060, we use a modest subset
python3 scripts/preprocess_fineweb.py \
  --limit 100000 \
  --sample-for-tokenizer 10000 \
  --min-token-count 32 \
  --context-length 2048 \
  --shard-size 100000000 \
  --vocab-size 50000 \
  --tokenizer simple \
  --output-dir datasets/fineweb_edu/processed/train
```

**Explanation:**
- `--limit 100000`: Process 100k documents (~500M-1B tokens)
- `--context-length 2048`: Use 2K context (fits in 12GB VRAM with batch_size=4)
- `--shard-size 100000000`: Create shards of ~100M tokens each
- `--vocab-size 50000`: Vocabulary size (fits in uint16)
- `--tokenizer simple`: Use lightweight tokenizer for speed

**Output:**
```
datasets/fineweb_edu/
└── processed/
    └── train/
        ├── metadata.json          # Dataset metadata
        ├── shard_000.bin         # Binary token shards
        ├── shard_001.bin
        ├── ...
        └── tokenizer/
            └── tokenizer.json
```

**Dataset size estimate:**
- 100k documents × ~5000 tokens average = ~500M tokens
- ~500M tokens × 2 bytes (uint16) = ~1GB total

### 2.2 Verify Dataset

```bash
python3 - <<'EOF'
import json
from pathlib import Path

metadata_path = Path("datasets/fineweb_edu/processed/train/metadata.json")
with open(metadata_path) as f:
    metadata = json.load(f)
    
print("Dataset metadata:")
for key, value in metadata.items():
    print(f"  {key}: {value}")
EOF
```

## Step 3: Model Configuration

Two model sizes are available:

### Small v1 (~90M parameters) - RECOMMENDED FOR RTX 3060

```
Hidden: 512
Layers: 16
Heads: 8
KV Heads: 2
FFN: 2048
Max Seq: 2048
```

Memory usage:
- Model weights: ~360MB
- Activations + gradients: ~8-10GB
- Optimizer states (FP32): ~1-2GB
- **Total: ~12GB** ✓

### Medium v1 (~251M parameters) - Requires gradient checkpointing

```
Hidden: 896
Layers: 24
Heads: 14
KV Heads: 4
FFN: 2432
Max Seq: 4096
```

Memory usage:
- Model weights: ~1GB
- Activations + gradients w/ checkpointing: ~8-10GB
- Optimizer states: ~3-4GB
- **Total: ~12GB** ⚠️ (requires gradient checkpointing)

## Step 4: Training

### 4.1 Train Small Model (Recommended)

```bash
python3 scripts/train_real.py \
  --data-dir datasets/fineweb_edu/processed/train \
  --model-size small \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --max-steps 100000 \
  --learning-rate 5e-4 \
  --checkpoint-dir checkpoints
```

**Configuration:**
- Batch size: 4 (fits in 12GB VRAM)
- Gradient accumulation: 4 steps (effective batch size = 16)
- Learning rate: 5e-4 (standard for LLM training)
- Checkpoints: Saved every 500 steps

**Expected training time:**
- ~500M tokens ÷ (4 batch × 2048 seq) = ~61k gradient updates
- On RTX 3060: ~1-2 weeks to 100k steps
- Adjust `--max-steps` as needed

### 4.2 Train Medium Model (With Gradient Checkpointing)

```bash
python3 scripts/train_real.py \
  --data-dir datasets/fineweb_edu/processed/train \
  --model-size medium \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --max-steps 50000 \
  --learning-rate 5e-4 \
  --context-length 2048 \
  --gradient-checkpointing \
  --checkpoint-dir checkpoints
```

### 4.3 Resume Training

```bash
# Training is automatic resumable from latest checkpoint
python3 scripts/train_real.py \
  --data-dir datasets/fineweb_edu/processed/train \
  --model-size small \
  --checkpoint-dir checkpoints
```

The trainer will:
1. Find the latest checkpoint in `checkpoints/model_step_*.pt`
2. Resume training from that step
3. Resume optimizer state (momentum, EMA, etc.)

## Step 5: Monitoring Training

### 5.1 View Training Logs

```bash
# Real-time logs
tail -f checkpoints/logs/history.jsonl

# Parse logs
python3 - <<'EOF'
import json
from pathlib import Path

with open("checkpoints/logs/history.jsonl") as f:
    for line in f:
        data = json.loads(line)
        if "step" in data:
            print(f"Step {data['step']}: loss={data['loss']:.4f}")
EOF
```

### 5.2 Check GPU Utilization

During training, in another terminal:

```bash
# Watch GPU memory and utilization
nvidia-smi -l 1

# Or with monitoring
watch -n 1 nvidia-smi
```

Expected output:
- **GPU Memory:** 11-12GB used
- **GPU Utilization:** 90-99% during training
- **Temp:** 60-80°C

## Step 6: Checkpoints and Recovery

### Checkpoint Structure

```
checkpoints/
├── model_step_500.pt      # Checkpoint at step 500
├── model_step_1000.pt     # Checkpoint at step 1000
└── logs/
    └── history.jsonl      # Training metrics
```

### Checkpoint Contents

Each `.pt` checkpoint contains:
- Model weights
- Optimizer state (Adam momentum + variance)
- Training step number

### Recovery Procedure

If training is interrupted:

```bash
# 1. Restart training (automatically resumes from latest)
python3 scripts/train_real.py \
  --data-dir datasets/fineweb_edu/processed/train \
  --model-size small \
  --checkpoint-dir checkpoints

# 2. If checkpoint is corrupted, delete it and restart
rm checkpoints/model_step_LATEST.pt
python3 scripts/train_real.py ...
```

## Step 7: Advanced Configuration

### 7.1 Gradient Checkpointing (Memory Optimization)

Reduces memory by ~40% but adds ~25% compute overhead:

```bash
python3 scripts/train_real.py \
  --model-size small \
  --gradient-checkpointing \
  --batch-size 8  # Can increase batch size
```

### 7.2 Larger Context Length

Requires smaller batch size:

```bash
python3 scripts/train_real.py \
  --context-length 4096 \
  --batch-size 2 \
  --gradient-accumulation-steps 8
```

### 7.3 Custom Learning Rate Schedule

Edit `scripts/train_real.py` to add warmup:

```python
# Placeholder for custom learning rate schedule
# To be added in next iteration
```

## Step 8: Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
python3 scripts/train_real.py --batch-size 2

# Or enable gradient checkpointing
python3 scripts/train_real.py --gradient-checkpointing

# Or reduce context length
python3 scripts/train_real.py --context-length 1024
```

### CUDA Errors

```bash
# Updates CUDA cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Then retry training
python3 scripts/train_real.py ...
```

### Slow Dataset Loading

If data loading is slow:
1. Verify shards are on fast disk (SSD)
2. Check disk I/O: `iotop`
3. Try smaller `shard_size` during preprocessing

### Loss Not Decreasing

1. Check learning rate: try `--learning-rate 1e-3` or `--learning-rate 1e-4`
2. Verify dataset: `python3 scripts/preprocess_fineweb.py --limit 100`
3. Check loss computation in logs

## Step 9: Next Steps

After training:

1. **Evaluation**
   ```bash
   python3 scripts/evaluate.py \
     --model-path checkpoints/model_step_10000.pt
   ```

2. **Generation**
   ```bash
   python3 scripts/generate.py \
     --model-path checkpoints/model_step_10000.pt \
     --prompt "The future of AI"
   ```

3. **Export Model**
   ```bash
   python3 scripts/export.py \
     --checkpoint-path checkpoints/model_step_10000.pt \
     --output-format safetensors
   ```

## Estimated Runtime

| Model | Context | Batch | Grad Accum | Steps | GPU Time | Wall Time |
|-------|---------|-------|-----------|-------|----------|-----------|
| Small | 2K      | 4     | 4         | 10K   | ~2 hours | ~3 hours  |
| Small | 2K      | 4     | 4         | 100K  | ~20 hours| ~24 hours |
| Medium| 2K      | 4     | 8         | 50K   | ~40 hours| ~48 hours |

## Support and Debugging

For issues:

1. Check GPU memory: `nvidia-smi`
2. Review error logs: `checkpoints/logs/history.jsonl`
3. Test on smaller dataset first: `--limit 100`
4. Verify CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`

## References

- FineWeb-Edu Dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- PyTorch Training: https://pytorch.org/docs/stable/notes/cuda.html
- Gradient Checkpointing: https://pytorch.org/docs/stable/checkpoint.html
