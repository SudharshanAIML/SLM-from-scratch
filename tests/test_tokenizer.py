import pytest

from slm.tokenizer.bpe_tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SPECIAL_IDS,
    UNK_ID,
    BPETokenizer,
)


def test_special_ids_are_the_documented_constants():
    assert (PAD_ID, UNK_ID, BOS_ID, EOS_ID) == (0, 1, 2, 3)
    assert SPECIAL_IDS == {"pad": 0, "unk": 1, "bos": 2, "eos": 3}


def test_trained_model_reports_the_same_special_ids(tokenizer):
    """The ids baked into the model must equal the ones the writer uses.

    This is the invariant whose absence let the old pipeline write a real word
    as its document separator.
    """
    assert tokenizer.sp.pad_id() == PAD_ID
    assert tokenizer.sp.unk_id() == UNK_ID
    assert tokenizer.sp.bos_id() == BOS_ID
    assert tokenizer.sp.eos_id() == EOS_ID
    assert tokenizer.eos_id == EOS_ID


def test_eos_id_is_not_an_ordinary_word(tokenizer):
    assert tokenizer.id_to_piece(EOS_ID) == "<eos>"
    assert tokenizer.id_to_piece(BOS_ID) == "<bos>"


def test_round_trip_preserves_text(tokenizer):
    text = "science and mathematics education."
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_vocab_size_is_exact(tokenizer):
    assert tokenizer.vocab_size == 512
    assert len(tokenizer) == 512


def test_unseen_text_still_encodes_without_unknown_tokens(tokenizer):
    """Subword BPE covers novel words by splitting them, unlike a word vocab."""
    ids = tokenizer.encode("quixotic zebras photosynthesize")
    assert ids
    assert UNK_ID not in ids


def test_bos_and_eos_are_added_on_request(tokenizer):
    ids = tokenizer.encode("history", add_bos=True, add_eos=True)
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID


def test_decode_skips_special_tokens_by_default(tokenizer):
    ids = tokenizer.encode("physics", add_bos=True, add_eos=True)
    assert tokenizer.decode(ids) == "physics"


def test_decode_reinserts_word_boundaries(tokenizer):
    """Regression: a word-level vocab joined with '' produced runtogethertext."""
    decoded = tokenizer.decode(tokenizer.encode("reading and writing"))
    assert decoded == "reading and writing"
    assert " " in decoded


def test_encode_batch_matches_individual_calls(tokenizer):
    texts = ["biology", "algebra and geometry"]
    assert tokenizer.encode_batch(texts) == [tokenizer.encode(t) for t in texts]


def test_ids_fit_the_uint16_shard_format(tokenizer):
    assert max(tokenizer.encode(" ".join(["chemistry", "network"] * 20))) < 65_536


def test_missing_model_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        BPETokenizer(tmp_path / "absent.model")
