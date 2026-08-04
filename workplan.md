# SLM from Scratch Workplan

## Completed milestones

- [x] Review the architecture and workflow documents.
- [x] Create a modular package structure for configs, tokenizer, data, layers, blocks, model, training, and inference.
- [x] Implement a minimal tokenizer and text preprocessing pipeline.
- [x] Implement a small transformer stack with RMSNorm, RoPE, attention, and SwiGLU.
- [x] Implement a simple training loop and example script.
- [x] Validate the pipeline by running a small training example.
- [x] Verify generation from the trained model with a prompt.

## Implemented modules

- `slm/configs/` for configuration dataclasses.
- `slm/tokenizer/simple_tokenizer.py` for a lightweight tokenizer.
- `slm/preprocessing/text_preprocessor.py` for cleaning input text.
- `slm/data/text_dataset.py` for training samples.
- `slm/layers/` for RMSNorm, RoPE, attention, SwiGLU, and MLP.
- `slm/blocks/transformer_block.py` for the transformer block.
- `slm/model/transformer.py` and `slm/model/generation.py` for the model and generation loop.
- `slm/training/trainer.py` for training.
- `slm/inference/sampler.py` for a simple inference wrapper.

## Next possible steps

1. Add a real SentencePiece tokenizer and dataset pipeline.
2. Support larger models and longer context windows.
3. Add checkpointing, logging, and evaluation metrics.
4. Add distributed training and mixed precision support.
5. Add inference optimizations such as KV-cache and sampling improvements.
