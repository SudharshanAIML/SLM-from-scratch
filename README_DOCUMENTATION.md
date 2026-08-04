# SLM Training Framework - Complete Documentation Index

Welcome! This is a production-ready SLM (Small Language Model) training framework optimized for NVIDIA RTX 3060 (12GB VRAM).

## 📚 Where to Start

**Are you completely new? Start here:**
→ [SETUP.md](SETUP.md) - Complete installation & first training guide (30 minutes)

**Are you experienced? Quick path:**
→ [QUICK_START.md](QUICK_START.md) - Condensed instructions for experts (5 minutes)

**Already training and need help?**
→ [TRAINING_MONITORING.md](TRAINING_MONITORING.md) - Real-time diagnostics & monitoring
→ [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common errors & fixes

**Want to understand the system?**
→ [ARCHITECTURE.md](ARCHITECTURE.md) - Deep dive into design & components

---

## 📖 Documentation Map

### For End Users

| Document | Purpose | Time | Audience |
|----------|---------|------|----------|
| [SETUP.md](SETUP.md) | Full installation & training walkthrough | 30 min | Everyone new |
| [QUICK_START.md](QUICK_START.md) | Minimal steps for experienced users | 5 min | Experts |
| [TRAINING_MONITORING.md](TRAINING_MONITORING.md) | Real-time training diagnostics & interpretation | 15 min | Active trainers |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Error diagnosis & resolution | 20 min | Debugging |
| [README.md](README.md) | Project overview | 5 min | Context |

### For Developers

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, components, extensibility |
| [slm/configs/model_config.py](slm/configs/model_config.py) | Model architecture definitions |
| [slm/model/transformer.py](slm/model/transformer.py) | Main transformer implementation |
| [slm/training/trainer.py](slm/training/trainer.py) | Training loop & checkpointing |

---

## 🚀 Quick Setup (3 Steps)

```bash
# 1. Setup environment (2 min)
python3 -m venv venv && source venv/bin/activate
pip install torch sentencepiece datasets pyyaml tqdm numpy --index-url https://download.pytorch.org/whl/cu118

# 2. Prepare data (5 min) 
python3 scripts/validate_setup.py  # Verify everything
python3 scripts/preprocess_fineweb.py --limit 100000  # Download & preprocess dataset

# 3. Start training (runs until stopped)
python3 scripts/train_real.py --data-dir datasets/fineweb_edu/processed/train
```

---

## 📊 Model Specifications

### Small v1 (~90M parameters) - RECOMMENDED

```
Architecture:
- Hidden size: 512
- Number of layers: 16
- Attention heads: 8 (GQA with 2 KV heads)
- FFN intermediate: 2048
- Max sequence length: 2048
- Vocabulary: 50,000

Hardware:
- GPU Memory: 11-12 GB (RTX 3060)
- Batch size: 4
- Gradient accumulation: 4
- Effective batch: 16
- Throughput: ~1000 tokens/sec
```

### Medium v1 (~251M parameters)

```
Architecture:
- Hidden size: 896
- Number of layers: 24
- Attention heads: 14 (GQA with 4 KV heads)
- FFN intermediate: 2432
- Max sequence length: 4096
- Vocabulary: 50,000

Hardware:
- GPU Memory: 11-12 GB (requires gradient checkpointing)
- Batch size: 4
- Gradient accumulation: 8
- Requires: --gradient-checkpointing flag
- Throughput: ~600-900 tokens/sec
```

---

## 📁 Project Structure

```
SLM-from-scratch/
├── 🔵 SETUP.md                   ← START HERE (new users)
├── 🔵 QUICK_START.md             ← START HERE (experienced)
├── 🔵 TRAINING_MONITORING.md     ← For active training
├── 🔵 TROUBLESHOOTING.md         ← For error solving
├── ARCHITECTURE.md               ← System deep dive
├── README.md                     ← Project overview
│
├── slm/                          # Main package
│   ├── model/                    # Transformer architecture
│   ├── configs/                  # Configuration system
│   ├── training/                 # Training loop & checkpointing
│   ├── data/                     # Data loading
│   ├── preprocessing/            # Dataset preparation
│   ├── tokenizer/                # Text tokenization
│   ├── layers/                   # Individual components
│   ├── blocks/                   # Composed modules
│   ├── inference/                # Generation utilities
│   └── evaluation/               # Metrics
│
├── scripts/                      # CLI entry points
│   ├── train_real.py             ← MAIN training script
│   ├── preprocess_fineweb.py     ← Data preprocessing
│   ├── train_example.py          ← Quick test example
│   └── validate_setup.py         ← Environment check
│
├── datasets/                     # Data storage (created during preprocessing)
├── checkpoints/                  # Model checkpoints (created during training)
└── tests/                        # Unit tests (optional)
```

---

## 🎯 Common Tasks

### Task: Start training for the first time
1. Complete [SETUP.md](SETUP.md) Step 1-3
2. Run: `python3 scripts/train_real.py --data-dir datasets/fineweb_edu/processed/train`

### Task: Monitor active training
1. See [TRAINING_MONITORING.md](TRAINING_MONITORING.md) "Real-Time Monitoring"
2. Run in separate terminal: `tail -f checkpoints/logs/history.jsonl | python3 -m json.tool`

### Task: Resume training after interruption
1. Just re-run the training command - it auto-detects checkpoint
2. `python3 scripts/train_real.py ...` (same flags)

### Task: Fix "out of memory" error
1. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) "Out of Memory"
2. Quick fix: `python3 scripts/train_real.py --batch-size 2 --gradient-checkpointing`

### Task: Use different model size
1. See [SETUP.md](SETUP.md) Step 3
2. Use: `python3 scripts/train_real.py --model-size medium --gradient-checkpointing`

### Task: Understand the codebase
1. Read [ARCHITECTURE.md](ARCHITECTURE.md)
2. Explore key files: `slm/model/transformer.py`, `slm/training/trainer.py`

### Task: Change learning rate or other hyperparameters
1. See [ARCHITECTURE.md](ARCHITECTURE.md) "Training Hyperparameter Defaults"
2. CLI flags: `--learning-rate 1e-3`, `--max-steps 50000`, etc.

---

## ✅ Validation Checklist

Before starting training, verify:

```bash
# Run automated validation
python3 scripts/validate_setup.py

# Manual checks:
☐ GPU available: nvidia-smi
☐ PyTorch installed: python3 -c "import torch; print(torch.cuda.is_available())"
☐ Project structure: ls slm/model/transformer.py
☐ Dataset ready: ls datasets/fineweb_edu/processed/train/shard_*.bin
```

---

## 📈 Training Timeline

### Small Model (90M params)

| Phase | Steps | GPU Time | Wall Time |
|-------|-------|----------|-----------|
| Initial phase (loss steep) | 1-5K | 1.5 hours | 2 hours |
| Rapid improvement | 5K-30K | 9 hours | 11 hours |
| Steady convergence | 30K-100K | 21 hours | 25 hours |

### Medium Model (251M params, with gradient checkpointing)

| Phase | Steps | GPU Time | Wall Time |
|-------|-------|----------|-----------|
| Initial phase | 1-2.5K | 4 hours | 5 hours |
| Rapid improvement | 2.5K-15K | 13 hours | 15 hours |
| Steady convergence | 15K-50K | 20 hours | 24 hours |

---

## 🔧 Key Commands Reference

```bash
# Environment
source venv/bin/activate
python3 scripts/validate_setup.py

# Preprocessing
python3 scripts/preprocess_fineweb.py --limit 100000

# Training
python3 scripts/train_real.py \
  --data-dir datasets/fineweb_edu/processed/train \
  --model-size small \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --max-steps 100000

# Training with gradient checkpointing (memory savings)
python3 scripts/train_real.py \
  --model-size medium \
  --gradient-checkpointing

# Monitor
tail -f checkpoints/logs/history.jsonl
nvidia-smi -l 1

# Resume (automatic)
python3 scripts/train_real.py ...  # Same command
```

---

## 📝 Key Concepts

### Gradient Accumulation
Accumulate gradients over multiple batches before updating weights. Allows larger effective batch size (batch_size × grad_accum_steps) while keeping per-batch memory low.

Example: batch_size=4, grad_accum=4 → effective batch=16

### Gradient Checkpointing
Trade compute for memory: recompute activations during backprop instead of storing them. Reduces memory by ~40%, adds ~25% compute overhead.

Use when: `--gradient-checkpointing`

### Context Length
Maximum sequence length tokens. Determines how much text the model sees at once.

Trade-off: Longer context → better modeling but higher memory
- RTX 3060: Use 2048 (default)

### Checkpoint
Saved state of model weights + optimizer during training. Enables resumption after interruption.

Saved every: 500 steps (configurable)
Location: `checkpoints/model_step_*.pt`

### Tokenizer
Converts text to token IDs (0-49999). Two options:
- SimpleTokenizer (fast, no training)
- SentencePiece (production, trained on dataset)

---

## 🐛 Debugging Essentials

### Training won't start
1. Run: `python3 scripts/validate_setup.py`
2. Check GPU: `nvidia-smi`
3. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

### GPU showing 0% utilization
1. Check disk I/O: `iotop -b`
2. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) "Training Very Slow"

### Loss not decreasing
1. Check logs: `tail checkpoints/logs/history.jsonl`
2. Try higher LR: `--learning-rate 1e-3`
3. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) "Loss Not Decreasing"

### Out of memory
1. Reduce batch: `--batch-size 2`
2. Enable checkpointing: `--gradient-checkpointing`
3. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) "Out of Memory"

---

## 📚 Additional Resources

### Papers Referenced
- [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GQA: Group Query Attention](https://arxiv.org/abs/2305.13245)
- [SwiGLU Activations](https://arxiv.org/abs/2002.05202)

### External Documentation
- [PyTorch CUDA Support](https://pytorch.org/docs/stable/cuda.html)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
- [SentencePiece](https://github.com/google/sentencepiece)

---

## 💡 Tips for Success

1. **Start with Small model** - Easier to debug, faster iteration
2. **Monitor GPU memory** - Run `nvidia-smi` in separate terminal
3. **Check first 100 steps** - Loss should decrease by 20-30%
4. **Save checkpoints frequently** - Default every 500 steps
5. **Use gradient checkpointing** - If memory pressure increases
6. **Read error messages carefully** - Most errors are self-explanatory

---

## 🆘 Getting Help

1. **Check docs:** Start with the document matching your task above
2. **Search troubleshooting:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md) covers 90% of issues
3. **System diagnostics:** Run `python3 scripts/validate_setup.py`
4. **Collect logs:** Share training logs + `nvidia-smi` output

---

## 📋 Next Steps

**If you haven't started yet:**
1. Read [SETUP.md](SETUP.md) (25 minutes)
2. Follow Step-by-Step (30 minutes total)
3. Begin training!

**If you're already training:**
1. Monitor with [TRAINING_MONITORING.md](TRAINING_MONITORING.md)
2. Check logs: `tail -f checkpoints/logs/history.jsonl`
3. Expected: Loss should decrease steadily

**If something's wrong:**
1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Run: `python3 scripts/validate_setup.py`
3. Usually fixable in 5 minutes

---

**Good luck with your training! 🚀**

Questions? Check the relevant guide above, or review [ARCHITECTURE.md](ARCHITECTURE.md) for deep understanding.
