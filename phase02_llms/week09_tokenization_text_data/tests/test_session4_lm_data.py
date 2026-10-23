"""Hand-written fixture checks for Session 4 (`lm_data.py`).

These fixtures are authored in `session4.md` and need no corpus and no network.
They are the executable form of the LM example contract. Run them with either:

    .venv/bin/python phase02_llms/week09_tokenization_text_data/tests/test_session4_lm_data.py
    .venv/bin/python -m pytest \
        phase02_llms/week09_tokenization_text_data/tests/test_session4_lm_data.py

Shapes are plain Python lists of lists, so no tensors are required. Fixture
stubs use `PAD = 0`, `EOS = 1`, and `IGNORE_INDEX = -100` from `lm_data.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

WEEK9_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK9_DIR))

import lm_data  # noqa: E402  (path insertion must happen first)

T = 3  # small block length for readable fixtures
EOS = 1
PAD = 0
IGNORE = -100


def test_full_block_needs_no_padding() -> None:
    """A stream of exactly T + 1 tokens yields one full, unpadded window."""
    windows, info = lm_data.build_source_windows([5, 6, 7, EOS], block_length=T)
    assert windows == [[5, 6, 7, EOS]]
    assert info["windows"] == 1
    assert info["final_window_padded"] is False
    assert info["lone_trailing_token"] is None

    batch = lm_data.make_lm_batch(windows, block_length=T)
    assert batch["input_ids"] == [[5, 6, 7]]
    assert batch["targets"] == [[6, 7, EOS]]
    assert batch["attention_mask"] == [[1, 1, 1]]
    assert batch["loss_mask"] == [[1, 1, 1]]


def test_short_final_window_is_right_padded() -> None:
    """session4.md's worked illustration: [5, 6, 1] with T = 3."""
    windows, info = lm_data.build_source_windows([5, 6, EOS], block_length=T)
    assert len(windows) == 1
    assert info["final_window_padded"] is True

    batch = lm_data.make_lm_batch(windows, block_length=T)
    assert batch["input_ids"] == [[5, 6, EOS]]
    assert batch["targets"] == [[6, EOS, IGNORE]]
    assert batch["attention_mask"] == [[1, 1, 1]]
    assert batch["loss_mask"] == [[1, 1, 0]]


def test_masks_are_not_the_same_object_rule() -> None:
    """The trap: a copied mask hides the padded position."""
    windows, _ = lm_data.build_source_windows([5, 6, EOS], block_length=T)
    batch = lm_data.make_lm_batch(windows, block_length=T)
    assert batch["loss_mask"] != batch["attention_mask"], (
        "loss_mask must exclude positions whose next token is padding"
    )


def test_real_eos_target_stays_trainable() -> None:
    """EOS must remain a valid target even when padding follows it."""
    windows, _ = lm_data.build_source_windows([5, EOS], block_length=T)
    batch = lm_data.make_lm_batch(windows, block_length=T)
    assert batch["input_ids"] == [[5, EOS, PAD]]
    assert batch["targets"] == [[EOS, IGNORE, IGNORE]]
    assert batch["attention_mask"] == [[1, 1, 0]]
    assert batch["loss_mask"] == [[1, 0, 0]], "the 5 -> EOS pair must stay trainable"


def test_lone_trailing_token_is_logged_not_windowed() -> None:
    """One leftover token has no valid pair, so it must not become a window."""
    windows, info = lm_data.build_source_windows([5], block_length=T)
    assert windows == []
    assert info["windows"] == 0
    assert info["lone_trailing_token"] == 5


def test_empty_stream_is_handled() -> None:
    windows, info = lm_data.build_source_windows([], block_length=T)
    assert windows == []
    assert info["windows"] == 0
    assert info["lone_trailing_token"] is None
    batch = lm_data.make_lm_batch(windows, block_length=T)
    assert batch["input_ids"] == []
    assert batch["targets"] == []
    assert batch["attention_mask"] == []
    assert batch["loss_mask"] == []


def test_consecutive_windows_share_exactly_one_token() -> None:
    stream = list(range(10, 20))
    windows, info = lm_data.build_source_windows(stream, block_length=T)
    assert len(windows) >= 2, "fixture needs at least two windows"
    for earlier, later in zip(windows, windows[1:]):
        assert earlier[-1] == later[0]
        assert len(earlier) == T + 1
        assert len(later) == T + 1
    assert info["windows"] == len(windows)


def test_targets_are_shifted_exactly_once() -> None:
    """For every window, target[i] equals the next source token, or IGNORE."""
    stream = list(range(100, 109))
    windows, _ = lm_data.build_source_windows(stream, block_length=T)
    batch = lm_data.make_lm_batch(windows, block_length=T)
    for source, inputs, targets in zip(windows, batch["input_ids"], batch["targets"]):
        for i in range(T):
            assert inputs[i] == source[i]
            expected = source[i + 1]
            if targets[i] == IGNORE:
                assert expected == PAD, "IGNORE is allowed only at padding"
            else:
                assert targets[i] == expected


def test_stream_appends_one_eos_per_record() -> None:
    """`build_split_stream` uses the real tokenizer path on a tiny corpus."""
    from tokenizer import train_tokenizer

    tokenizer = train_tokenizer(["alpha beta", "gamma"], lm_data.BLOCK_LENGTH)
    records = ["alpha beta", "gamma", ""]
    stream = lm_data.build_split_stream(tokenizer, records)

    nonempty = [record for record in records if record.strip()]
    encodings = [tokenizer.encode(r, add_special_tokens=False).ids for r in nonempty]
    assert len(stream) == sum(len(ids) for ids in encodings) + len(nonempty)
    assert stream == lm_data.build_split_stream(tokenizer, records), "not deterministic"

    cursor = 0
    for ids in encodings:
        cursor += len(ids)
        assert stream[cursor] == EOS, "each record must be closed by one EOS"
        cursor += 1


FIXTURE_CHECKS = [
    test_full_block_needs_no_padding,
    test_short_final_window_is_right_padded,
    test_masks_are_not_the_same_object_rule,
    test_real_eos_target_stays_trainable,
    test_lone_trailing_token_is_logged_not_windowed,
    test_empty_stream_is_handled,
    test_consecutive_windows_share_exactly_one_token,
    test_targets_are_shifted_exactly_once,
    test_stream_appends_one_eos_per_record,
]


def main() -> None:
    passed = 0
    for check in FIXTURE_CHECKS:
        check()
        passed += 1
        print(f"PASS {check.__name__}")
    print(f"\n{passed}/{len(FIXTURE_CHECKS)} fixture checks passed")


if __name__ == "__main__":
    main()
