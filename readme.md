# final architecture model
| Component           | Final Choice                              | Why                           |
| ------------------- | ----------------------------------------- | ----------------------------- |
| Parameters          | ~251M                                     | Fits your target              |
| Layers              | **24**                                    | Best depth/width balance      |
| Hidden Size         | **896**                                   | Efficient and within budget   |
| Head Dimension      | **64**                                    | Tensor Core friendly          |
| Query Heads         | **14**                                    | Exactly matches hidden size   |
| KV Heads            | **4**                                     | Efficient GQA                 |
| FFN                 | **2432**                                  | SwiGLU with ~8/3 expansion    |
| Vocabulary          | **50K**                                   | Good coverage                 |
| Context             | **4096** (curriculum from 1024→2048→4096) | Faster training               |
| Norm                | RMSNorm                                   | Modern standard               |
| Positional Encoding | RoPE                                      | Standard                      |
| Attention           | FlashAttention (SDPA fallback)            | Efficient                     |
| Precision           | BF16 (FP16 fallback if needed)            | Stable                        |
| Weight Tying        | Yes                                       | Saves ~45M parameters         |
| Bias Terms          | **Disabled**                              | Common in modern decoder LLMs |



Final Architecture (SLM v1.0)
                     Input Tokens
                           │
                    SentencePiece Tokenizer
                           │
                     Token Embedding
                           │
                    (Weight Tied Later)
                           │
        ┌────────────────────────────────────┐
        │        Transformer Block × N       │
        └────────────────────────────────────┘
                           │
                       RMSNorm
                           │
                     Linear LM Head
                           │
                        Softmax

Every transformer block will be:

                    Input
                      │
                  RMSNorm
                      │
          QKV Linear Projection
                      │
                    RoPE
                      │
          FlashAttention + GQA
                      │
             Output Projection
                      │
                Residual Add
                      │
                  RMSNorm
                      │
             SwiGLU FeedForward
                      │
                Residual Add
Component Selection
1. Normalization

✅ RMSNorm

Reason

Faster than LayerNorm
Fewer operations
Used in Llama, Gemma, Qwen, DeepSeek, Mistral
2. Position Encoding

✅ RoPE

We'll implement RoPE ourselves.

Future support:

YaRN scaling
NTK scaling
LongRoPE

No learned positional embeddings.

3. Attention

Definitely

✅ GQA

Not MHA.

Example

32 Query Heads

8 KV Heads

instead of

32 Q

32 K

32 V

This reduces KV cache by 75%.

4. Attention Backend

Use

FlashAttention v2

when CUDA supports it.

Otherwise

torch.nn.functional.scaled_dot_product_attention()

Modern PyTorch already dispatches to FlashAttention kernels when available, so we can design the code with a backend abstraction rather than hard-coding a single implementation.

5. Feed Forward

I'd use

SwiGLU

Activation

SiLU

not GELU.

Reason

SwiGLU literally means

SiLU(xW₁) ⊙ (xW₂)

followed by the output projection.

Modern models (Llama, Gemma, Mistral, Qwen) all use SiLU inside SwiGLU. Using GELU would change it into a different gated MLP variant (often called GEGLU), which is also valid but less common today.

6. Precision

I would not train in FP16.

I'd choose

BF16

Reasons

Much larger exponent range than FP16
More numerically stable
Fewer gradient overflows
Usually no loss scaling required
Supported on A100, H100, RTX 4090, and newer accelerators

Training:

Weights → BF16

Activations → BF16

Optimizer states → FP32

Gradients → BF16

This is essentially the standard recipe for modern LLM training.

7. Optimizer

AdamW

Parameters

β1 = 0.9

β2 = 0.95

Weight Decay = 0.1

These values are commonly used for decoder-only LLMs, though we'll tune them later if needed.

8. Learning Rate Schedule
Warmup

↓

Cosine Decay

↓

Minimum LR
9. Weight Initialization

Instead of vanilla Xavier

I'd use

Scaled Normal Initialization

following modern Transformer practices, with residual projections scaled appropriately to improve stability in deeper networks.

10. Embedding

Tie

Input Embedding

=

Output Embedding

Benefits

saves millions of parameters
usually improves perplexity
11. KV Cache

Must support

Prefill

↓

Decode

↓

Continuous Generation

This is essential for efficient inference.

12. Causal Mask

Dynamic.

No fixed-size masks.

13. Dropout

Training

Attention Dropout

MLP Dropout

Residual Dropout

Inference

Disabled

For large-scale pretraining, many recent LLMs use little or no dropout. We can keep it configurable and default it to 0.0 unless experiments suggest otherwise.

14. Loss

Cross Entropy

with

Shifted Labels

Exactly as GPT/Llama.

15. Gradient Checkpointing

Enabled.

Allows much larger batch sizes.

16. Gradient Clipping
1.0
17. Tensor Layout

Use

Batch

↓

Sequence

↓

Hidden

Everywhere.

Avoid unnecessary transposes.

Complete Transformer Block
Input
 │
 ▼
RMSNorm
 │
 ▼
Linear(QKV)
 │
 ▼
RoPE
 │
 ▼
FlashAttention (GQA)
 │
 ▼
Linear
 │
 ▼
Residual
 │
 ▼
RMSNorm
 │
 ▼
Linear
 │
 ▼
SiLU
 │
 ├─────────────┐
 │             │
 ▼             ▼
Gate        Value
 │             │
 └──────⊙──────┘
        │
        ▼
Output Linear
        │
        ▼
Residual
Software Architecture

Rather than one large model.py, I'd organize it like this:

slm/
│
├── config/
│   ├── model_config.py
│   ├── train_config.py
│   └── tokenizer_config.py
│
├── tokenizer/
│
├── data/
│
├── layers/
│   ├── embedding.py
│   ├── rmsnorm.py
│   ├── rope.py
│   ├── attention.py
│   ├── flash_attention.py
│   ├── gqa.py
│   ├── swiglu.py
│   └── mlp.py
│
├── blocks/
│   └── transformer_block.py
│
├── model/
│   ├── transformer.py
│   └── generation.py
│
├── training/
│   ├── trainer.py
│   ├── optimizer.py
│   ├── scheduler.py
│   └── checkpoint.py
│
├── inference/
│   ├── kv_cache.py
│   ├── sampler.py
│   └── beam.py
│
├── evaluation/
│
└── utils/
One addition I strongly recommend

Since this is a long-term project, let's make it configuration-driven. Every architectural choice should be switchable from a config file.

attention: gqa
attention_backend: flash_attention
norm: rmsnorm
activation: swiglu
rope: true
rope_scaling: yarn
precision: bf16
optimizer: adamw
scheduler: cosine
dropout: 0.0
tie_embeddings: true
gradient_checkpointing: true

With this design, the same codebase can reproduce GPT-style MHA, Llama-style GQA, or future experiments by changing configuration rather than rewriting modules.