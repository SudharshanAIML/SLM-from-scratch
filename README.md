# SLM from Scratch — 259M Decoder-Only Transformer

A small language model built from first principles in PyTorch. No `transformers`,
no `accelerate` — the tokenizer, data format, attention, training loop and
sampler are all in this repo.

```
Token IDs        Embedding          Transformer Block x20            LM Head
 [2048]  ─────►  [32,768 x 1024] ─► RMSNorm → GQA+RoPE → +res  ─────► [32,768 logits]
                                    RMSNorm → SwiGLU   → +res
                 33.554M                 225.485M                    tied (0 new)
```

## Specification

| | |
|---|---|
| Parameters | **259,039,232** (+1,024 final norm = 259,040,256) |
| Layers | 20 |
| Hidden size | 1024 |
| Query heads | 16 |
| KV heads | 4 (GQA, group size 4) |
| Head dim | 64 |
| FFN (intermediate) | 2816 |
| Vocab | 32,768 (SentencePiece BPE, byte fallback) |
| Context | 2048 |

Normalization RMSNorm · Positions RoPE · Attention GQA via SDPA · FFN SwiGLU
· Weight tying on · Biases off · Precision BF16/FP16 on CUDA, FP32 on CPU

Verify the arithmetic against the running code at any time:

```bash
python scripts/inspect_model.py --build
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate

# CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# or CUDA 12.x
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The same code runs on both. `device` and `precision` default to `auto`:
CUDA with BF16 where supported, FP16 where not, FP32 on CPU.

## Use

**1 — Build the dataset.** Streams FineWeb-Edu, trains a 32,768-piece BPE
model, and packs tokens into memory-mappable `uint16` shards with a train/val
split:

```bash
python scripts/preprocess.py --limit 100000 --vocab-size 32768
```

**2 — Train.**

```bash
python scripts/train.py --data-dir datasets/fineweb_edu --batch-size 8
python scripts/train.py --resume                     # continue from latest
```

Reduce memory with `--gradient-checkpointing` and a smaller `--batch-size`,
raising `--gradient-accumulation-steps` to hold the tokens-per-step constant.
Cadence is tunable with `--log-every`, `--eval-every`, `--eval-steps`,
`--save-every` and `--keep-last-n-checkpoints`.

To exercise the whole path on a CPU without training 259M parameters, swap in
the tiny preset — same code, same dataflow, ~0.1M params:

```bash
python scripts/train.py --model debug --device cpu --context-length 64 \
    --batch-size 4 --max-steps 40 --log-every 10
```

**3 — Generate.**

```bash
python scripts/generate.py --prompt "The history of science" --temperature 0.8
```

**Tests.**

```bash
python -m pytest
```

## Layout

```
slm/
  configs/       ModelConfig (the 259M spec) and TrainConfig
  tokenizer/     SentencePiece BPE with fixed special ids
  data/          shard writer + memory-mapped IterableDataset
  preprocessing/ FineWeb-Edu streaming pipeline
  layers/        RMSNorm, RoPE, GQA attention, SwiGLU
  blocks/        the transformer block
  model/         the model and KV-cached generation
  training/      trainer, LR schedule, checkpoints, logging
  evaluation/    loss and perplexity
scripts/         preprocess.py, train.py, generate.py, inspect_model.py
tests/           86 tests covering every stage
```

## Data format

Shards are headerless little-endian `uint16` arrays, so training memory-maps
them with zero parsing. Documents are packed continuously:

```
<bos> tokens… <eos> <bos> tokens… <eos> …
```

Special ids are fixed in one place (`slm/tokenizer/bpe_tokenizer.py`) and
asserted on load: `pad=0, unk=1, bos=2, eos=3`. Each shard directory carries a
`metadata.json` with the vocab size, the EOS id actually written, token and
document counts, and a SHA-256 per shard. Training reads its vocabulary from
that metadata rather than a constant, so the embedding table and the token ids
cannot drift apart.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full parameter derivation and
component-level detail.
