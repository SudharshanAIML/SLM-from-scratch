# Architecture & Codebase Guide

Complete documentation of the SLM training framework architecture.

## Project Structure

```
SLM-from-scratch/
│
├── SETUP.md                          # Installation & training guide (START HERE)
├── QUICK_START.md                    # For experienced users
├── TRAINING_MONITORING.md            # Real-time training diagnostics
├── TROUBLESHOOTING.md                # Common errors and solutions
├── README.md                         # Project overview
│
├── slm/                              # Main package
│   ├── __init__.py
│   │
│   ├── configs/                      # Configuration management
│   │   ├── __init__.py
│   │   ├── model_config.py           # Model architecture params
│   │   ├── train_config.py           # Training hyperparams
│   │   └── tokenizer_config.py       # Tokenizer settings
│   │
│   ├── model/                        # Transformer implementation
│   │   ├── __init__.py
│   │   ├── transformer.py            # Main transformer + factory methods
│   │   └── generation.py             # Inference utilities
│   │
│   ├── layers/                       # Individual components
│   │   ├── attention.py              # Multi-head & Group Query Attention
│   │   ├── rope.py                   # Rotary Positional Embeddings
│   │   ├── rmsnorm.py                # Root Mean Square Normalization
│   │   ├── mlp.py                    # Feed-forward networks
│   │   └── swiglu.py                 # SwiGLU activation
│   │
│   ├── blocks/                       # Composed modules
│   │   └── transformer_block.py      # Attention + MLP + Normalization
│   │
│   ├── tokenizer/                    # Text tokenization
│   │   ├── simple_tokenizer.py       # Lightweight word-based tokenizer
│   │   └── sentencepiece_tokenizer.py # Production SentencePiece integration
│   │
│   ├── data/                         # Data loading
│   │   ├── text_dataset.py           # Raw text dataset
│   │   └── binary_dataset.py         # Memory-mapped binary shards
│   │
│   ├── preprocessing/                # Data preparation
│   │   ├── text_preprocessor.py      # Text cleaning
│   │   └── fineweb_preprocessor.py   # FineWeb-Edu → binary shards
│   │
│   ├── training/                     # Training infrastructure
│   │   ├── trainer.py                # Main training loop with checkpointing
│   │   ├── checkpoint.py             # Checkpoint save/load
│   │   └── logger.py                 # Metric logging
│   │
│   ├── inference/                    # Deployment
│   │   ├── kv_cache.py               # KV-cache for fast generation
│   │   └── sampler.py                # Sampling strategies
│   │
│   └── evaluation/                   # Metrics
│       ├── metrics.py                # General metrics
│       └── perplexity.py             # Perplexity evaluation
│
├── scripts/                          # CLI entry points
│   ├── preprocess_fineweb.py        # Dataset preprocessing CLI
│   ├── train_real.py                 # Training CLI (MAIN ENTRY POINT)
│   ├── train_example.py              # Quick test example
│   └── validate_setup.py             # Environment validation
│
├── datasets/                         # Data storage
│   └── fineweb_edu/
│       └── processed/train/
│           ├── metadata.json         # Dataset metadata
│           ├── shard_000.bin         # Binary token shards (uint16)
│           └── tokenizer/
│               ├── tokenizer.json
│               └── tokenizer.model
│
├── checkpoints/                      # Training artifacts
│   ├── model_step_500.pt             # Saved model + optimizer state
│   └── logs/
│       └── history.jsonl             # Training metrics log
│
└── tests/                            # Unit tests (optional)
```

## Core Components

### 1. Configuration System (`slm/configs/`)

**Purpose:** Centralized, version-controlled hyperparameters

**Key Files:**

- **`model_config.py`** - Architecture definition
  ```python
  class ModelConfig:
      small_v1()    # ~90M params, 512 hidden, 16 layers
      medium_v1()   # ~251M params, 896 hidden, 24 layers
  ```

- **`train_config.py`** - Training hyperparameters
  ```python
  class TrainConfig:
      rtx3060_v1()  # Optimized for RTX 3060 with gradient accumulation
  ```

**Design Pattern:**
- Dataclasses for type safety
- Factory methods for standard configs
- Easy duplication for experiments

### 2. Tokenizer System (`slm/tokenizer/`)

**Two implementations available:**

1. **SimpleTokenizer** - Lightweight, fast
   - Character or word-level tokenization
   - No training required
   - ~50K vocabulary

2. **SentencePieceTokenizer** - Production-ready
   - Subword tokenization (BPE)
   - Trained on dataset
   - Better compression

**Pipeline:**
```
Text → Tokenizer.encode() → Token IDs [0-49999]
Token IDs → Tokenizer.decode() → Text
```

### 3. Data Pipeline (`slm/data/`, `slm/preprocessing/`)

**Processing Flow:**

```
HuggingFace FineWeb-Edu
    ↓
FineWebPreprocessor.iter_documents()  # Stream docs
    ↓
Tokenizer.encode_documents()          # Tokenize
    ↓
Pack tokens (with EOS padding)        # Sequences
    ↓
Save as uint16 binary shards          # 100M+ tokens each
    ↓
BinaryShardDataset (np.memmap)        # Memory-mapped loader
    ↓
Training loop (batch_size × seq_len)
```

**Key Design Decisions:**

- **Streaming:** Don't load full dataset to RAM
- **Binary sharding:** Faster I/O than JSON/JSONL
- **uint16 format:** Fits 50k vocab, 2x faster than uint32
- **Memmapping:** Load shards on-demand, not all at once

### 4. Model Architecture (`slm/model/`, `slm/layers/`, `slm/blocks/`)

**Core Components:**

```
Transformer (slm/model/transformer.py)
├── Token Embedding (50K vocab)
├── N × TransformerBlock (slm/blocks/transformer_block.py)
│   ├── GroupQueryAttention (slm/layers/attention.py)
│   │   ├── RoPE (slm/layers/rope.py)
│   │   └── KV Cache Support
│   ├── MLP (slm/layers/mlp.py) + SwiGLU (slm/layers/swiglu.py)
│   ├── RMSNorm (slm/layers/rmsnorm.py)
│   └── Residual Connections
└── Output Head (50K vocab logits)
```

**Key Optimizations:**

1. **Group Query Attention (GQA):** 
   - Reduces KV cache memory by 2-4x
   - Minimal quality loss

2. **Rotary Positional Embeddings (RoPE):**
   - Better extrapolation than absolute positions
   - No explicit position indices

3. **SwiGLU Activation:**
   - Improved gradient flow
   - Better than GELU on smaller models

4. **RMSNorm:**
   - Simpler than LayerNorm
   - Better numerical stability

### 5. Training System (`slm/training/`)

**Core Loop (`trainer.py`):**

```python
for step in range(max_steps):
    # Gradient accumulation loop
    for accum_step in range(gradient_accumulation_steps):
        batch = dataset.get_batch()
        output = model(batch)
        loss = compute_loss(output, batch)
        loss.backward()
    
    # Update weights
    optimizer.step()
    optimizer.zero_grad()
    
    # Checkpoint (every save_every steps)
    if step % save_every == 0:
        save_checkpoint(model, optimizer, step)
    
    # Log metrics
    logger.log_step(step, loss, lr, grad_norm)
```

**Key Features:**

- **Gradient Accumulation:** Effective batch = batch_size × grad_accum_steps
- **Gradient Checkpointing:** Save 40% memory at 25% compute cost
- **Automatic Resumption:** Detects and loads latest checkpoint
- **Logging:** Real-time metrics to `history.jsonl`

### 6. Entry Points (`scripts/`)

**`train_real.py` - Main Training CLI**

```bash
python3 scripts/train_real.py \
  --data-dir datasets/fineweb_edu/processed/train \
  --model-size small \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --max-steps 100000
```

**`preprocess_fineweb.py` - Dataset Preparation**

```bash
python3 scripts/preprocess_fineweb.py \
  --limit 100000 \
  --vocab-size 50000 \
  --context-length 2048
```

**`validate_setup.py` - Environment Check**

```bash
python3 scripts/validate_setup.py
```

## Data Flow Diagram

### Training Pipeline

```
1. PREPROCESSING
   ├─ Download FineWeb-Edu (streaming)
   ├─ Build tokenizer vocabulary
   ├─ Tokenize documents
   ├─ Pack tokens (EOS-padded)
   └─ Save binary shards (uint16)

2. LOADING
   ├─ Load metadata.json (vocab, token count)
   ├─ Open shard files with np.memmap
   └─ Slice windows for training samples

3. TRAINING ITERATION
   ├─ Fetch batch[batch_size, seq_len]
   ├─ Forward pass → logits[batch_size, seq_len, vocab]
   ├─ Compute loss (cross-entropy)
   ├─ Backward pass
   ├─ Accumulate gradients (grad_accum_steps times)
   ├─ Optimizer step
   ├─ Clear gradients
   ├─ Log metrics
   └─ Save checkpoint (every 500 steps)

4. CHECKPOINTING
   ├─ Save model weights
   ├─ Save optimizer state (momentum, variance)
   ├─ Save step number
   └─ Append to history.jsonl

5. RESUMPTION
   ├─ Find latest checkpoint
   ├─ Load model & optimizer state
   └─ Continue from step N
```

## Memory Layout on RTX 3060

**Small Model Configuration (90M params):**

```
GPU Memory Budget: 12 GB

Forward Pass:
├─ Model weights: 360 MB
├─ Activations: 3 GB
└─ Attention buffers: 1 GB
                Total: ~4.4 GB

Backward Pass:
├─ Gradient buffers: 360 MB
├─ Activation gradients: 3 GB
└─ Optimizer states (Adam): 2 × 360 MB
                Total: ~7.5 GB

Peak Memory: ~11 GB ✓
```

**With Gradient Checkpointing:**
- Reduces activation storage from 3 GB → 1.5 GB
- Recomputes activations during backprop (+25% compute)
- Peak memory: ~8.5 GB
- Allows batch_size=8 with grad_accum=4

## Configuration Factory Pattern

### Small Model (RTX 3060 Optimized)

```python
config = ModelConfig.small_v1(vocab_size=50000)
# Hidden: 512, Layers: 16, Heads: 8, KV Heads: 2
# ~90M params, fits in 12GB with gradient accumulation
```

### Medium Model (Requires Gradient Checkpointing)

```python
config = ModelConfig.medium_v1(vocab_size=50000)
# Hidden: 896, Layers: 24, Heads: 14, KV Heads: 4
# ~251M params, requires gradient checkpointing
```

## Training Hyperparameter Defaults

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `batch_size` | 4 | Fits in 12GB VRAM |
| `gradient_accumulation_steps` | 4 | Effective batch = 16 |
| `learning_rate` | 5e-4 | Standard LLM training |
| `max_grad_norm` | 1.0 | Prevent gradient explosion |
| `save_every` | 500 | Checkpoint frequently for recovery |
| `context_length` | 2048 | Fits 4 sequences × 2K = 8K tokens/batch |

## Extensibility Points

### 1. Add New Model Architectures

**File:** `slm/configs/model_config.py`

```python
class ModelConfig:
    @staticmethod
    def large_v1(vocab_size: int, max_seq_len: int = 4096):
        # Add large model config here
        return ModelConfig(...)
```

### 2. Add New Training Strategies

**File:** `slm/training/trainer.py`

```python
class Trainer:
    def _training_loop_with_warmup(self, ...):
        # Add warmup schedule
        pass
```

### 3. Add New Layers

**File:** `slm/layers/`

```python
# Create new_layer.py
class NewLayer(nn.Module):
    def forward(self, x): ...
```

Then import in `slm/model/transformer.py`

### 4. Add Evaluation Metrics

**File:** `slm/evaluation/`

```python
def compute_accuracy(predictions, targets):
    return (predictions == targets).mean()
```

## Performance Characteristics

### Small Model (90M params)

| Metric | Value |
|--------|-------|
| Training throughput | 800-1200 tokens/sec |
| Peak memory | 11-12 GB |
| Time for 100k steps | 24-48 hours |
| Loss(step 1) | ~5.5 |
| Loss(step 10k) | ~2.5-3.5 |

### Medium Model (251M params)

With gradient checkpointing:

| Metric | Value |
|--------|-------|
| Training throughput | 600-900 tokens/sec |
| Peak memory | 11-12 GB |
| Time for 50k steps | 24-40 hours |
| Compute overhead | +25% due to recomputation |

## Common Modifications

### Change Model Size

```python
# In train_real.py
config = ModelConfig.medium_v1()  # Switch to medium
```

### Change Learning Rate Schedule

```python
# In trainer.py _training_loop()
lr = args.learning_rate * (1 - step / max_steps)  # Linear decay
optimizer.param_groups[0]['lr'] = lr
```

### Add Warmup

```python
# In trainer.py
warmup_steps = 1000
if step < warmup_steps:
    lr = args.learning_rate * (step / warmup_steps)
```

### Use Mixed Precision

```bash
# Enable in train_real.py
python3 scripts/train_real.py --use-mixed-precision
```

Then wrap forward pass:
```python
with torch.autocast('cuda', dtype=torch.bfloat16):
    output = model(batch)
```

## Testing & Validation

### Unit Tests

Run individual components:

```bash
python3 << 'EOF'
import torch
from slm.model.transformer import Transformer
from slm.configs.model_config import ModelConfig

config = ModelConfig.small_v1(vocab_size=50000)
model = Transformer(config)
batch = torch.randint(0, 50000, (4, 2048))
output = model(batch)
print(f"Output shape: {output.shape}")  # [4, 2048, 50000]
EOF
```

### Integration Tests

Quick end-to-end test:

```bash
# Preprocess minimal dataset
python3 scripts/preprocess_fineweb.py --limit 10

# Train for 10 steps
python3 scripts/train_real.py --max-steps 10 --batch-size 2
```

## Debugging & Profiling

### Profile Training

```bash
python3 -m cProfile -s cumulative scripts/train_real.py --max-steps 10
```

### Memory Profiling

```bash
python3 -m memory_profiler scripts/train_real.py
```

### Inspect Model

```python
from slm.model.transformer import Transformer
from slm.configs.model_config import ModelConfig

config = ModelConfig.small_v1()
model = Transformer(config)
print(model)  # Architecture overview
print(sum(p.numel() for p in model.parameters()) / 1e6, "M params")
```

## Checkpoint File Format

**Structure of `model_step_N.pt`:**

```python
{
    'step': int,                      # Training step number
    'model_state': dict,              # Model weight tensors
    'optimizer_state': dict,          # Optimizer momentum + variance
    'config': ModelConfig,            # Architecture config
}
```

**Loading checkpoint:**

```python
checkpoint = torch.load("model_step_1000.pt", map_location="cuda")
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
step = checkpoint['step']
```

## References

- **RoPE:** https://arxiv.org/abs/2104.09864
- **GQA:** https://arxiv.org/abs/2305.13245
- **SwiGLU:** https://arxiv.org/abs/2002.05202
- **Gradient Checkpointing:** https://pytorch.org/docs/stable/checkpoint.html
