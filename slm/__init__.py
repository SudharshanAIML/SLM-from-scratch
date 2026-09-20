"""A 259M-parameter decoder-only small language model, built from scratch."""

from slm.configs import ModelConfig, TrainConfig
from slm.model import Transformer, generate
from slm.tokenizer import BPETokenizer

__all__ = ["ModelConfig", "TrainConfig", "Transformer", "generate", "BPETokenizer"]
__version__ = "2.0.0"
