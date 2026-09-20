from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import sentencepiece as spm

# Single source of truth for special token ids.
#
# These match the ids handed to SentencePieceTrainer below, so the id written
# into a binary shard is always the id the tokenizer itself reports. Keeping
# these in one place is what prevents the classic "the EOS separator is
# actually a real word" corruption.
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

SPECIAL_IDS = {"pad": PAD_ID, "unk": UNK_ID, "bos": BOS_ID, "eos": EOS_ID}


class BPETokenizer:
    """SentencePiece BPE tokenizer with fixed special-token ids.

    Trained with byte fallback, so every input encodes losslessly and no
    token ever maps to <unk>.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._sp: spm.SentencePieceProcessor | None = None
        self.model_path: Path | None = None
        if model_path is not None:
            self.load(model_path)

    # ------------------------------------------------------------------ train

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        model_prefix: str | Path,
        vocab_size: int = 32_768,
        model_type: str = "bpe",
        character_coverage: float = 0.9995,
        max_sentence_length: int = 16_384,
        input_sentence_size: int = 2_000_000,
        num_threads: int = 16,
    ) -> "BPETokenizer":
        """Train a BPE model, streaming the corpus to disk rather than buffering it.

        `texts` is consumed lazily, so a multi-gigabyte stream trains in
        roughly constant memory.
        """
        # 256 byte-fallback pieces + 4 specials are reserved before any
        # learned merge, so anything near that size cannot train.
        if vocab_size < 300:
            raise ValueError(
                f"vocab_size={vocab_size} is too small for byte fallback; "
                "use at least 300"
            )

        model_prefix = Path(model_prefix)
        model_prefix.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as handle:
            corpus_path = Path(handle.name)
            lines = 0
            for text in texts:
                # SentencePiece reads one sentence per line; newlines inside a
                # document would otherwise split it into unrelated samples.
                flat = " ".join(text.split())
                if flat:
                    handle.write(flat + "\n")
                    lines += 1

        if lines == 0:
            corpus_path.unlink(missing_ok=True)
            raise ValueError("No training text was provided to the tokenizer")

        try:
            spm.SentencePieceTrainer.train(
                input=str(corpus_path),
                model_prefix=str(model_prefix),
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=character_coverage,
                max_sentence_length=max_sentence_length,
                input_sentence_size=input_sentence_size,
                shuffle_input_sentence=True,
                num_threads=max(1, num_threads),
                pad_id=PAD_ID,
                unk_id=UNK_ID,
                bos_id=BOS_ID,
                eos_id=EOS_ID,
                pad_piece="<pad>",
                unk_piece="<unk>",
                bos_piece="<bos>",
                eos_piece="<eos>",
                normalization_rule_name="nmt_nfkc",
                # Any character the model never saw decomposes into raw
                # byte pieces instead of <unk>, so the corpus has no OOV.
                byte_fallback=True,
            )
        finally:
            corpus_path.unlink(missing_ok=True)

        return cls(model_prefix.with_suffix(".model"))

    # ------------------------------------------------------------------- load

    def load(self, model_path: str | Path) -> "BPETokenizer":
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer model not found: {path}")
        self._sp = spm.SentencePieceProcessor(model_file=str(path))
        self.model_path = path
        self._verify_special_ids()
        return self

    def _verify_special_ids(self) -> None:
        """Fail loudly if a loaded model disagrees with the shard-format ids."""
        assert self._sp is not None
        actual = {
            "pad": self._sp.pad_id(),
            "unk": self._sp.unk_id(),
            "bos": self._sp.bos_id(),
            "eos": self._sp.eos_id(),
        }
        if actual != SPECIAL_IDS:
            raise ValueError(
                f"Tokenizer special ids {actual} do not match the expected "
                f"{SPECIAL_IDS}. Shards written with this model would carry "
                "mismatched document separators."
            )

    # --------------------------------------------------------------- encoding

    @property
    def sp(self) -> spm.SentencePieceProcessor:
        if self._sp is None:
            raise RuntimeError("Tokenizer model is not loaded")
        return self._sp

    @property
    def vocab_size(self) -> int:
        return int(self.sp.get_piece_size())

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def encode_batch(
        self, texts: Sequence[str], add_bos: bool = False, add_eos: bool = False
    ) -> list[list[int]]:
        return [self.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts]

    def decode(self, token_ids: Sequence[int], skip_special: bool = True) -> str:
        ids = list(token_ids)
        if skip_special:
            ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
        return self.sp.decode(ids)

    def id_to_piece(self, token_id: int) -> str:
        return self.sp.id_to_piece(int(token_id))

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={self.vocab_size}, path={self.model_path})"
