import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from corpus import split_examples
from tokenizer import EOS_TOKEN, PAD_TOKEN, CharVocabulary, build_sequence_tensors


def test_corpus_split_produces_readme_examples():
    examples = split_examples(PROJECT_DIR / "README.md")
    assert examples
    assert any("Week 5" in example for example in examples)


def test_vocabulary_round_trip_and_special_tokens():
    vocab = CharVocabulary.from_sentences(["abc"])
    ids = vocab.encode_text("cab")

    assert PAD_TOKEN in vocab.token_to_id
    assert EOS_TOKEN in vocab.token_to_id
    assert vocab.decode_ids(ids) == "cab"


def test_sequence_tensors_have_expected_shape_and_eos_target():
    vocab = CharVocabulary.from_sentences(["abc"])
    X, y, _, skipped = build_sequence_tensors(["abc"], vocabulary=vocab, sequence_length=5)

    assert skipped == 0
    assert X.shape == (1, 5)
    assert y.shape == (1, 5)
    assert y[0, 2].item() == vocab.eos_id
    assert y[0, 3].item() == vocab.pad_id
