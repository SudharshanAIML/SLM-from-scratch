# Architecture — 259M Decoder-Only SLM

Every number below is produced by the code. Run `python scripts/inspect_model.py --build`
to print this derivation and confirm it against an instantiated model.

## Model configuration

```
Parameters   259,039,232  (+1,024 with the final RMSNorm)
Layers       20
Hidden       1024
Q Heads      16
KV Heads     4
Head Dim     64
FFN          2816
Vocab        32,768
Context      2048
```

`hidden_size = num_heads × head_dim` → `1024 = 16 × 64`. The config rejects any
combination that breaks this, along with odd head dims (RoPE needs pairs) and
head counts not divisible by the KV head count (GQA needs whole groups).

## Data flow

```
FineWeb-Edu (HF streaming)
        │
        ▼
  document filter          min 32 words, non-empty text
        │
        ▼
  SentencePiece BPE        32,768 pieces, byte fallback
        │
        ▼
  packing                  <bos> doc <eos> <bos> doc <eos> …
        │
        ▼
  uint16 binary shards     headerless, memory-mapped
        │
        ▼
  BinaryShardDataset       (x, y) = (tokens[i:i+L], tokens[i+1:i+L+1])
        │
        ▼
  Transformer              → (batch, seq, 32768) logits → cross-entropy
```

Training never touches parquet. The corpus is converted once; every epoch
afterwards is a memory-mapped read.

## Forward pass

```
input_ids [B, T]
    │
    ▼
Token Embedding  32,768 × 1024 ─────────────── 33,554,432
    │
    ▼
┌─ Transformer Block ─────────────────────────┐
│  x ──► RMSNorm ──► GQA + RoPE ──► + ────►   │  ×20
│  └──────────────────────────────┘           │
│  x ──► RMSNorm ──► SwiGLU     ──► + ────►   │
│  └──────────────────────────────┘           │
└─────────────────────────────────────────────┘
    │
    ▼
Final RMSNorm ───────────────────────────────── 1,024
    │
    ▼
LM Head (tied with embedding) ───────────────── 0 new
    │
    ▼
logits [B, T, 32,768]
```

## Components

### RMSNorm — 1,024 params per instance

```
RMS(x) = sqrt( (1/H) Σ xᵢ² + ε )
y      = x / RMS(x) ⊙ γ
```

Only the gain `γ` is learned. The reduction runs in fp32 even under BF16
autocast — computing a mean of squares in half precision loses enough accuracy
to destabilise a 20-layer stack. Two per block plus one final: `2 × 1024 = 2,048`
per layer.

### RoPE — 0 params

```
q, k ──► RoPE ──► attention
```

Rotates query/key pairs by a position-dependent angle, so the attention score
`⟨R(q,m), R(k,n)⟩` depends only on the relative distance `m − n`. The cos/sin
tables are a function of position alone, so they are built once and sliced —
not rebuilt per layer per step — and are registered non-persistently so they
never enter a checkpoint.

### Grouped-Query Attention — 2,621,440 params per layer

```
              x
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
    Q         K         V
 16 heads  4 heads   4 heads
    │         │         │
    └───── RoPE(Q,K) ───┘
              │
        Attention(Q,K,V) = softmax(QKᵀ / √64) V
              │
              ▼
        Output Projection
```

| Projection | Shape | Params |
|---|---|---|
| `q_proj` | 1024 × 1024 | 1,048,576 |
| `k_proj` | 1024 × 256 | 262,144 |
| `v_proj` | 1024 × 256 | 262,144 |
| `o_proj` | 1024 × 1024 | 1,048,576 |
| | **total** | **2,621,440** |

4 KV heads instead of 16 shrink the K and V matrices to a quarter, and shrink
the inference KV cache by the same factor. Each KV head is shared by 4 query
heads; the expansion is a stride trick, not a copy.

### SwiGLU — 8,650,752 params per layer

```
           x
         /   \
   gate_proj  up_proj
        │       │
      SiLU      │
        └── × ──┘
            │
        down_proj
```

```
y = down( SiLU(gate(x)) ⊙ up(x) )
P = 3 × H × F = 3 × 1024 × 2816 = 8,650,752
```

Three matrices, no biases — the single largest parameter block in the model, at
77% of each layer.

## Parameter accounting

```
Embedding
  32,768 × 1,024                        =  33,554,432

One transformer block
  GQA attention                         =   2,621,440
  SwiGLU                                =   8,650,752
  RMSNorm × 2                           =       2,048
  ────────────────────────────────────────────────────
  per layer                             =  11,274,240

  × 20 layers                           = 225,484,800
  ────────────────────────────────────────────────────
  225,484,800 + 33,554,432              = 259,039,232

                                        ≈ 259M PARAMETERS
```

The LM head adds nothing: it shares storage with the embedding. Including the
standalone final RMSNorm, the instantiated model reports **259,040,256**.

## Training

**Initialization.** All weights `N(0, 0.02)`; the two projections that write
into the residual stream (`o_proj`, `down_proj`) are scaled by `1/√(2·20)` so
residual variance stays roughly constant with depth. A correctly initialised
model starts at a loss of about `ln(32,768) ≈ 10.4` — `tests/test_model.py`
asserts this.

**Schedule.** Linear warmup then cosine decay to `min_learning_rate`. One
iteration of the outer loop is exactly one optimizer step, so the step counter,
the LR schedule, checkpoint filenames and resume all count the same thing.

**Optimizer.** AdamW, β = (0.9, 0.95). Weight decay applies to 2-D matmul
weights only; RMSNorm gains are 1-D and are excluded.

**Precision.** BF16 autocast on capable CUDA devices, FP16 with a grad scaler
otherwise, FP32 on CPU. Gradients are unscaled before clipping, so
`max_grad_norm` means the same thing in every mode.

**Checkpoints.** Written atomically via a temp file and rename, pruned to the
last N, and listed by parsed step number — sorting by filename would place
`model_step_1000` before `model_step_999`.

## Inference

Generation runs through the KV cache: the prompt is prefilled once, then each
new token attends to cached keys and values, making decoding linear in length
rather than quadratic. Cached keys are stored already-rotated and are never
re-rotated; new queries and keys receive RoPE at their true absolute position.
`tests/test_attention.py` asserts that cached decoding reproduces a full
forward pass exactly.

Sampling supports temperature (0 = greedy), top-k, top-p and a repetition
penalty, stops at `<eos>`, and halts at the context limit rather than raising.
