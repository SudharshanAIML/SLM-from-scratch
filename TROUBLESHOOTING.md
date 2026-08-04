# Troubleshooting Guide

Detailed solutions for common issues during setup and training.

## Environment Setup Issues

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Error Message:**
```
ModuleNotFoundError: No module named 'torch'
```

**Causes:**
- PyTorch not installed
- Virtual environment not activated
- Wrong Python version

**Solutions:**

1. Activate virtual environment:
```bash
source venv/bin/activate
```

2. Reinstall PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Verify installation:
```bash
python3 -c "import torch; print(torch.__version__)"
```

### Issue: "CUDA is not available" or GPU not detected

**Error Message:**
```
torch.cuda.is_available() returns False
CUDA is not available even though you have an NVIDIA GPU
```

**Causes:**
- NVIDIA driver not installed
- CUDA toolkit not installed
- PyTorch compiled for different CUDA version
- Wrong PyTorch version for your CUDA

**Solutions:**

1. Check NVIDIA driver:
```bash
nvidia-smi
```
If command not found, install driver:
```bash
# Ubuntu
sudo apt install nvidia-driver-520  # or your driver version

# Verify
nvidia-smi
```

2. Check CUDA version from driver output:
```bash
# From nvidia-smi output, look for "CUDA Version"
nvidia-smi | grep "CUDA Version"
```

3. Reinstall PyTorch for your CUDA version:
```bash
# For CUDA 11.8 (most compatible)
pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1 (if you prefer)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. Verify CUDA integration:
```bash
python3 << 'EOF'
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"Device Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
EOF
```

### Issue: "RuntimeError: Found GPU with CUDA Compute Capability..."

**Error Message:**
```
RuntimeError: Found GPU with CUDA Compute Capability X.X.
```

**Causes:**
- PyTorch version doesn't support RTX 3060 architecture
- CUDA version mismatch

**Solution:**
```bash
# RTX 3060 requires CUDA 11.0+
# Use CUDA 11.8 (most stable)
pip uninstall -y torch torchvision torchaudio
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.0 \
  --index-url https://download.pytorch.org/whl/cu118
```

## Data Preprocessing Issues

### Issue: "FileNotFoundError: Can't find 'fineweb-edu' dataset"

**Error Message:**
```
FileNotFoundError: Can't find 'fineweb-edu' on Hugging Face
```

**Causes:**
- Internet connection issue
- Dataset name typo
- Hugging Face credentials required

**Solutions:**

1. Check internet connection:
```bash
curl https://huggingface.co -I
```

2. Verify dataset exists:
```bash
python3 << 'EOF'
from datasets import load_dataset
# Lists all FineWeb datasets
dataset = load_dataset("HuggingFaceFW/fineweb-edu", streaming=True, split="sample-10BT")
print("Dataset loaded successfully")
EOF
```

3. If blocked by rate limit, use cache:
```bash
python3 scripts/preprocess_fineweb.py \
  --limit 1000 \
  --cache-dir ./hf_cache  # Cache locally
```

### Issue: "ValueError: Vocab size too small"

**Error Message:**
```
ValueError: SentencePiece requires at least X documents
ValueError: Not enough unique characters for vocab size Y
```

**Causes:**
- `--vocab-size` too large for sample
- `--sample-for-tokenizer` too small

**Solution:**

Use smaller vocab size:
```bash
# Start with 10k vocab instead of 50k
python3 scripts/preprocess_fineweb.py \
  --limit 100000 \
  --vocab-size 10000 \
  --sample-for-tokenizer 50000  # More samples for vocab
```

### Issue: "MemoryError: Unable to allocate memory"

**Error Message:**
```
MemoryError: Unable to allocate X.XX GiB for an array
```

**Causes:**
- Preprocessing loading entire dataset to RAM
- `--limit` too high
- System RAM becoming full

**Solutions:**

1. Reduce dataset size:
```bash
python3 scripts/preprocess_fineweb.py --limit 10000
```

2. Use streaming mode (already default):
```bash
python3 scripts/preprocess_fineweb.py \
  --limit 100000 \
  --shard-size 50000000  # Smaller shards = less RAM needed
```

3. Monitor RAM during preprocessing:
```bash
watch -n 1 free -h
```

### Issue: "Connection timeout" during dataset download

**Error Message:**
```
ConnectionError: (Max retries exceeded with url)
```

**Causes:**
- Network timeout during download
- Transient network issue

**Solutions:**

1. Increase timeout:
```bash
python3 scripts/preprocess_fineweb.py \
  --limit 100000 \
  --timeout 60  # If flag exists
```

2. Retry download:
```bash
# Dataset will resume from cached partial data
python3 scripts/preprocess_fineweb.py --limit 100000
```

3. Use smaller sample:
```bash
# Download less data, faster
python3 scripts/preprocess_fineweb.py --limit 10000
```

## Training (Runtime) Issues

### Issue: "CUDA out of memory" during training

**Error Message:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Causes:**
- Batch size too large
- Context length too large
- Model doesn't fit in 12GB VRAM
- GPU memory leak (unbounded accumulation)

**Solutions (in order of effectiveness):**

1. **Immediately reduce batch size:**
```bash
python3 scripts/train_real.py \
  --batch-size 2 \
  --gradient-accumulation-steps 8  # Keep effective batch = 16
```

2. **Enable gradient checkpointing** (saves 40% memory, adds 25% compute):
```bash
python3 scripts/train_real.py \
  --batch-size 4 \
  --gradient-checkpointing
```

3. **Reduce context length** (fewer tokens per batch):
```bash
python3 scripts/train_real.py \
  --context-length 1024  # From 2048
```

4. **Clear CUDA cache before restarting:**
```bash
python3 -c "import torch; torch.cuda.empty_cache()"
python3 scripts/train_real.py ...
```

5. **Verify no background GPU processes:**
```bash
nvidia-smi | grep python
# Kill any orphaned processes:
pkill -9 python3
```

6. **Last resort - use CPU (very slow):**
```bash
python3 scripts/train_real.py --device cpu  # Not recommended
```

### Issue: "Loss not decreasing" after 1000 steps

**Symptoms:**
```
Step 100: loss = 5.2
Step 500: loss = 5.1
Step 1000: loss = 5.0  # No significant change
```

**Causes:**
- Learning rate too low
- Bad data quality
- Hyperparameter issue

**Solutions:**

1. **Increase learning rate** (most common):
```bash
python3 scripts/train_real.py \
  --learning-rate 1e-3  # From 5e-4
```

2. **Check data quality:**
```python
# Verify data is not corrupted
python3 << 'EOF'
import json
from pathlib import Path

metadata_file = Path("datasets/fineweb_edu/processed/train/metadata.json")
with open(metadata_file) as f:
    metadata = json.load(f)
    print(f"Dataset tokens: {metadata.get('total_tokens', 'unknown')}")
    print(f"Documents: {metadata.get('total_documents', 'unknown')}")
    print(f"Vocab size: {metadata.get('vocab_size', 'unknown')}")
EOF
```

3. **Try warmup schedule** (modify train_real.py):
```python
# Add to training loop
warmup_steps = 1000
if step < warmup_steps:
    lr = args.learning_rate * (step / warmup_steps)
```

4. **Verify gradient flow:**
```bash
# Add to training loop temporarily
print(f"Gradient norm: {torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))}")
```

### Issue: "Loss becomes NaN" during training

**Symptoms:**
```
Step 100: loss = 4.5
Step 150: loss = 2.1
Step 200: loss = NaN  # Diverged!
```

**Causes:**
- Learning rate too high
- Exploding gradients
- Numerical instability

**Solutions:**

1. **Immediately reduce learning rate:**
```bash
python3 scripts/train_real.py --learning-rate 1e-4
```

2. **Lower context length** (fewer tokens = easier training):
```bash
python3 scripts/train_real.py --context-length 1024
```

3. **Use smaller batch size** (more stable gradients):
```bash
python3 scripts/train_real.py --batch-size 2
```

4. **Check for NaN in checkpoint:**
```python
python3 << 'EOF'
import torch
from pathlib import Path

checkpoint_file = max(Path("checkpoints").glob("model_step_*.pt"))
checkpoint = torch.load(checkpoint_file)
model_state = checkpoint['model_state']

has_nan = False
for name, param in model_state.items():
    if torch.isnan(param).any():
        print(f"NaN in {name}")
        has_nan = True

if not has_nan:
    print("No NaN found in model")
EOF
```

### Issue: "Training very slow" or GPU utilization low

**Symptoms:**
```
nvidia-smi: GPU Util = 20-40% (should be 90%+)
Training speed: 100 tokens/sec (should be 800+)
```

**Causes:**
- Data loading bottleneck
- CPU-GPU synchronization
- Small batch size

**Diagnostic:**
```bash
# Monitor GPU and CPU
watch -n 1 "nvidia-smi | grep Volatile; top -bn1 | grep -E 'Cpu|%Mem'"
```

**Solutions:**

1. **Check if data loading is bottleneck:**
```bash
# Monitor disk I/O during training
iotop -b -n 1
```

   If disk I/O is high:
   - Move dataset to faster SSD
   - Reduce number of parallel workers

2. **Increase batch size** (if not OOM):
```bash
python3 scripts/train_real.py --batch-size 8 --gradient-accumulation-steps 2
```

3. **Check for excessive logging:**
```bash
# Logging every 10 steps is fine, every step is slow
# Modify train_real.py if needed
```

4. **Verify no background processes:**
```bash
ps aux | grep python
# Kill unrelated processes
```

### Issue: "Checkpoint loading fails"

**Error Message:**
```
RuntimeError: Unexpected key in state_dict when loading checkpoint
```

**Causes:**
- Checkpoint corrupted
- Loading wrong checkpoint
- Model config mismatch

**Solutions:**

1. **Delete corrupted checkpoint and restart:**
```bash
# Find latest checkpoint
ls -lt checkpoints/model_step_*.pt | head -3

# Delete latest (may be corrupted)
rm checkpoints/model_step_LATEST.pt

# Restart - will resume from previous checkpoint
python3 scripts/train_real.py ...
```

2. **Verify checkpoint integrity:**
```python
python3 << 'EOF'
import torch
try:
    checkpoint = torch.load("checkpoints/model_step_1000.pt")
    print("Checkpoint keys:", checkpoint.keys())
    print("Model state keys:", len(checkpoint['model_state']))
    print("✓ Checkpoint OK")
except Exception as e:
    print(f"✗ Checkpoint corrupted: {e}")
EOF
```

3. **Clear all checkpoints and start from scratch:**
```bash
rm -rf checkpoints/model_step_*.pt
python3 scripts/train_real.py ...  # Starts fresh training
```

### Issue: "Process terminated" (no error message)

**Symptoms:**
```
Training running...
[suddenly exits]
```

**Causes:**
- Out of memory (kernel OOM killer)
- Disk full
- Timeout

**Diagnosis:**
```bash
# Check system logs for OOM
dmesg | tail -50 | grep -i "killed"

# Check disk space
df -h

# Check if process still running
ps aux | grep train
```

**Solution:**

1. **If disk full:**
```bash
# Delete old checkpoints
rm checkpoints/model_step_{100,200,300,400}.pt

# Or reduce shard size
python3 scripts/preprocess_fineweb.py --shard-size 50000000
```

2. **If OOM killer:**
   - Reduce batch size and restart
   - Enable gradient checkpointing

3. **If timeout:**
   - Run with tmux or screen for persistence
   ```bash
   tmux new-session -d -s training python3 scripts/train_real.py ...
   tmux attach -t training
   ```

## Validation and Testing

### Quick Health Check

```bash
python3 scripts/validate_setup.py
```

### Test Training with Small Dataset

```bash
# Only download 100 documents for quick testing
python3 scripts/preprocess_fineweb.py --limit 100

# Run 10 training steps
python3 scripts/train_real.py \
  --data-dir datasets/fineweb_edu/processed/train \
  --max-steps 10 \
  --batch-size 2  # Conservative
```

Expected output (for small test):
- Training should complete in <1 minute
- Loss should decrease (e.g., 5.5 → 5.0)
- No memory errors

## Getting Help

If none of these solutions work:

1. **Collect diagnostic info:**
```bash
# Save all relevant info
{
  echo "=== PyTorch ===" 
  python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  echo "=== GPU ==="
  nvidia-smi
  echo "=== CPU/Memory ==="
  free -h
  df -h
  echo "=== Dataset ==="
  ls -lh datasets/fineweb_edu/processed/train/
  echo "=== Training Log ==="
  tail -20 checkpoints/logs/history.jsonl
} > diagnostic.txt
```

2. **Reduce to minimal reproduction:**
   - Preprocess only 100 documents
   - Train for only 10 steps
   - Use smallest batch size

3. **Check online resources:**
   - PyTorch CUDA issues: https://pytorch.org/docs/stable/cuda.html
   - Training metrics: See [TRAINING_MONITORING.md](TRAINING_MONITORING.md)
