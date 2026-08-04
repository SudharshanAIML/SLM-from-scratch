# Training Monitoring and Troubleshooting Guide

## Real-Time Monitoring During Training

### 1. Key Metrics to Watch

Each training step logs the following to `checkpoints/logs/history.jsonl`:

```json
{
  "step": 100,
  "loss": 4.253,
  "tokens_per_sec": 1024,
  "lr": 0.0005,
  "grad_norm": 1.234
}
```

### 2. Interpreting Loss Curves

**Expected behavior:**
```
Step 1-100:    Loss: 5.0-6.0  (random initialization)
Step 100-500:  Loss: 4.0-5.0  (rapid decrease)
Step 500+:     Loss: 2.0-4.0  (steady decline)
```

**Red flags:**
- Loss unchanged for 500+ steps → learning rate too low
- Loss diverging (increasing) → learning rate too high
- Loss oscillating wildly → unstable gradient accumulation

**Quick fix:**

```bash
# If loss not decreasing, try different learning rates
for lr in 1e-4 5e-4 1e-3; do
  echo "Testing LR: $lr"
  python3 scripts/train_real.py --learning-rate $lr --max-steps 1000 &
done
```

### 3. Performance Metrics

**Throughput:** tokens/sec should be:
- With batch_size=4, gradient_accum=4: **800-1200 tokens/sec**
- With gradient_checkpointing: **600-900 tokens/sec**

**Memory:** nvidia-smi should show:
- **11-12GB used** (near max)
- **90-99% utilization**

**Convergence:** After first 1000 steps:
- Loss should drop by ~30-40%
- After 10k steps: loss should be ~50-60% of initial

### 4. Real-Time Monitoring Commands

**Monitor in separate terminal:**

```bash
# Option 1: Simple histogram
watch -n 5 'tail -20 checkpoints/logs/history.jsonl | tail -1 | python3 -m json.tool'

# Option 2: Plot with numpy
python3 - <<'EOF'
import json
import statistics
from pathlib import Path

with open("checkpoints/logs/history.jsonl") as f:
    losses = []
    for line in f:
        losses.append(json.loads(line)["loss"])
    
    if losses:
        last_100 = losses[-100:]
        avg = statistics.mean(last_100)
        std = statistics.stdev(last_100) if len(last_100) > 1 else 0
        print(f"Last 100 steps: avg_loss={avg:.4f} ± {std:.4f}")
        print(f"Latest loss: {losses[-1]:.4f}")
        print(f"Initial loss: {losses[0]:.4f}")
        print(f"Improvement: {(1 - losses[-1]/losses[0])*100:.1f}%")
EOF

# Option 3: GPU memory and utilization
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature_gpu \
  --format=csv,noheader -l 1
```

## Checkpoint Management

### Check Checkpoint Status

```bash
# List checkpoints
ls -lh checkpoints/model_step_*.pt

# Check file size (✓ normal: 1-2GB)
du -h checkpoints/model_step_*.pt | tail -5

# Latest checkpoint
ls -lt checkpoints/model_step_*.pt | head -1
```

### Inspect Checkpoint Contents

```bash
python3 - <<'EOF'
import torch
import json
from pathlib import Path

# Find latest checkpoint
checkpoint_dir = Path("checkpoints")
latest = max(checkpoint_dir.glob("model_step_*.pt"))

print(f"Loading: {latest}")
checkpoint = torch.load(latest, map_location="cpu")

print("\nCheckpoint keys:", checkpoint.keys())
print(f"Step: {checkpoint.get('step', 'N/A')}")
print(f"Model state dict keys: {len(checkpoint['model_state'])}")
print(f"Optimizer state dict keys: {len(checkpoint['optimizer_state'])}")

# Check model size
total_params = sum(p.numel() for p in checkpoint['model_state'].values() if isinstance(p, torch.Tensor))
print(f"Total parameters: {total_params/1e6:.1f}M")

# Check for NaN
for key, val in checkpoint['model_state'].items():
    if isinstance(val, torch.Tensor) and torch.isnan(val).any():
        print(f"⚠️ NaN found in {key}")
EOF
```

### Delete Old Checkpoints (To Save Space)

```bash
# Keep only last 5 checkpoints
ls -1t checkpoints/model_step_*.pt | tail -n +6 | xargs rm

# Or delete specific
rm checkpoints/model_step_5000.pt
```

## Advanced Diagnostics

### 1. Loss Spike Detection

If you notice sudden loss spikes:

```bash
# Check if gradient is exploding
python3 - <<'EOF'
import json
from pathlib import Path

with open("checkpoints/logs/history.jsonl") as f:
    prev_loss = None
    for i, line in enumerate(f):
        data = json.loads(line)
        if prev_loss and data['loss'] > prev_loss * 1.5:
            print(f"⚠️ Loss spike at step {data['step']}: {prev_loss:.4f} → {data['loss']:.4f}")
        prev_loss = data['loss']
EOF
```

**Actions:**
- Reduce learning rate: `--learning-rate 2.5e-4`
- Use gradient norm clipping: already enabled (default 1.0)
- Use mixed precision: `--use-mixed-precision` (if not already)

### 2. Batch Effect Analysis

Compare different batch sizes:

```bash
# Run 100 steps with different configs
for bs in 2 4 8; do
  rm -rf test_checkpoint_${bs}
  python3 scripts/train_real.py \
    --batch-size $bs \
    --max-steps 100 \
    --checkpoint-dir test_checkpoint_${bs} &
done
wait

# Compare velocities
for bs in 2 4 8; do
  echo "Batch size: $bs"
  tail -1 test_checkpoint_${bs}/logs/history.jsonl | python3 -c "import sys, json; print(json.load(sys.stdin))"
done
```

### 3. Data Quality Check

```bash
# Verify no corrupted shards
python3 - <<'EOF'
import numpy as np
from pathlib import Path

shard_dir = Path("datasets/fineweb_edu/processed/train")
for shard_file in sorted(shard_dir.glob("shard_*.bin")):
    print(f"Checking {shard_file.name}...")
    try:
        data = np.memmap(shard_file, dtype=np.uint16, mode='r')
        print(f"  ✓ {len(data)} tokens")
        
        # Check for valid token IDs (should be < 50000)
        if (data >= 50000).any():
            print(f"  ⚠️ Found invalid tokens > 50000")
    except Exception as e:
        print(f"  ✗ Error: {e}")
EOF
```

## Common Issues and Solutions

| Issue | Symptom | Cause | Solution |
|-------|---------|-------|----------|
| OOM | "CUDA out of memory" | Batch too large | Reduce `--batch-size` or enable `--gradient-checkpointing` |
| Stalled training | GPU util 0% | Dataset loading slow | Move shards to SSD, check disk I/O |
| Slow training | GPU util 20-50% | Small batch, bottleneck | Increase `--batch-size` or parallel preprocessing |
| Loss exploding | Loss → inf | Gradient explosion | Reduce `--learning-rate`, lower `--context-length` |
| Loss flat | No decrease for 1k+ steps | Learning rate too low | Increase `--learning-rate` to 1e-3 |
| Checkpoint corrupt | Load error after crash | Disk error during save | Delete `.pt` file, restart training |

## Performance Scaling

### Memory vs Batch Size

```bash
# Measure memory for different batch sizes
for bs in 1 2 4 8; do
  echo "Batch size: $bs"
  python3 -c "
import torch
from slm.configs import ModelConfig, TrainConfig
from slm.model import Transformer

config = ModelConfig.small_v1(vocab_size=50000)
model = Transformer(config)
torch.cuda.reset_peak_memory_stats()

batch = torch.randint(0, 1000, (${bs}, 2048)).to('cuda')
model(batch.to('cuda'))

print(f'  Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB')
"
done
```

### Throughput Scaling

Expected throughput by config:

```
Small model, batch=4, grad_accum=4:  ~1000 tokens/sec
Small model, batch=8, grad_accum=4:  ~1500 tokens/sec (OOM risk)
Small model w/ checkpointing:        ~800 tokens/sec
Medium model w/ checkpointing:       ~600 tokens/sec
```

## Resuming After Interruption

### Check Last Successful Step

```bash
python3 - <<'EOF'
import json
from pathlib import Path

history_file = Path("checkpoints/logs/history.jsonl")
if history_file.exists():
    with open(history_file) as f:
        lines = f.readlines()
    last_entry = json.loads(lines[-1])
    print(f"Last successful step: {last_entry['step']}")
    print(f"Last loss: {last_entry['loss']:.4f}")
else:
    print("No training history found")
EOF
```

### Safe Resume

```bash
# Verify latest checkpoint exists
latest_ckpt=$(ls -t checkpoints/model_step_*.pt 2>/dev/null | head -1)

if [ -z "$latest_ckpt" ]; then
    echo "No checkpoint found. Starting fresh."
else
    echo "Resuming from $latest_ckpt"
fi

# Resume training (auto-detects checkpoint)
python3 scripts/train_real.py \
  --data-dir datasets/fineweb_edu/processed/train \
  --model-size small \
  --checkpoint-dir checkpoints
```

## Memory Profiling

### During Training

```bash
# In separate terminal, watch memory growth
python3 - <<'EOF'
import torch
import time
from datetime import datetime

for i in range(60):  # Run for 60 seconds
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Allocated: {mem:.2f}GB, Reserved: {mem_reserved:.2f}GB")
    time.sleep(1)
EOF
```

### Peak Memory Usage

```bash
python3 - <<'EOF'
import torch
torch.cuda.reset_peak_memory_stats()

# Run one training step...
# Then check:
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
EOF
```

## Log Analysis Dashboard

Save this as `monitor.py` for real-time monitoring:

```python
#!/usr/bin/env python3
import json
import time
from pathlib import Path
from collections import deque

history_file = Path("checkpoints/logs/history.jsonl")
last_mtime = 0
recent_losses = deque(maxlen=100)

while True:
    if history_file.exists():
        current_mtime = history_file.stat().st_mtime
        if current_mtime > last_mtime:
            with open(history_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        recent_losses.append(entry['loss'])
                    except:
                        pass
            last_mtime = current_mtime
    
    if recent_losses:
        avg_loss = sum(recent_losses) / len(recent_losses)
        min_loss = min(recent_losses)
        max_loss = max(recent_losses)
        latest = recent_losses[-1]
        
        print(f"\r[{time.strftime('%H:%M:%S')}] Latest: {latest:.4f} | Avg: {avg_loss:.4f} | Min: {min_loss:.4f} | Max: {max_loss:.4f}", end="")
    
    time.sleep(1)
```

Run with: `python3 monitor.py`

## Conclusion

Training should be stable and show consistent loss decrease. Monitor the first 100 steps carefully—if loss doesn't decrease, adjust learning rate immediately.
