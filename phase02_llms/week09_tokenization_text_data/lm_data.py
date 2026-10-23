"""Week 9 Session 4: split-specific LM chunks, targets, and masks.

LEARNER SCAFFOLD. The constants, signatures, and docstrings below define the
contract recorded in `session4.md`; the bodies are yours to implement. Session 2
started the same way. No solution is provided here.

Run from the repository root once implemented:

    .venv/bin/python phase02_llms/week09_tokenization_text_data/lm_data.py
"""

from __future__ import annotations

from tokenizers import Tokenizer

# Frozen settings (see session4.md "The Contract To Implement").
BLOCK_LENGTH = 128
STRIDE = BLOCK_LENGTH
FIXTURE_BATCH_SIZE = 2
PAD_ID = 0
EOS_ID = 1
IGNORE_INDEX = -100


def build_split_stream(
    tokenizer: Tokenizer, rows: list[str], eos_id: int = EOS_ID
) -> list[int]:
    """Encode nonempty records in stable order and append one EOS per record.

    Return one continuous token stream for this split. Packing may cross row
    boundaries inside the split; it must never cross into another split.
    """
    stream: list[int] = []
    for row in rows:
        if not row.strip():
            continue
        stream.extend(tokenizer.encode(row, add_special_tokens=False).ids)
        stream.append(eos_id)
    return stream


def build_source_windows(
    stream: list[int], block_length: int = BLOCK_LENGTH
) -> tuple[list[list[int]], dict]:
    """Cut `block_length + 1` token windows at stride `block_length`.

    Right-pad the final short window only when it holds at least one valid
    input/target pair. Return the windows plus a small record describing the
    padding decision and any lone trailing token that has no target.

    Window `k` is `stream[k * block_length : k * block_length + block_length + 1]`.
    The returned record must expose these keys:

        stream_length          - len(stream)
        windows                - number of windows returned
        final_window_padded    - True when the last window was right-padded
        lone_trailing_token    - the leftover token with no target, else None
    """
    windows: list[list[int]] = []
    final_window_padded = False
    start = 0

    # A slice needs at least one (input, target) pair, so two tokens, to be
    # worth scoring. Window k spans [k*T, k*T + T], so window k+1 starts on
    # window k's last token; stop rather than re-slicing an already-used token.
    while len(stream) - start >= 2:
        window = stream[start : start + block_length + 1]
        if len(window) < block_length + 1:
            final_window_padded = True
            window = window + [PAD_ID] * (block_length + 1 - len(window))
        windows.append(window)
        start += block_length

    # Every token a window touches is consumed; only a 1-token stream is left.
    covered = 0 if not windows else min(len(stream), start + 1)
    lone_trailing_token: int | None = (
        stream[-1] if len(stream) - covered == 1 else None
    )

    info = {
        "stream_length": len(stream),
        "windows": len(windows),
        "final_window_padded": final_window_padded,
        "lone_trailing_token": lone_trailing_token,
    }
    return windows, info


def make_lm_batch(
    source_windows: list[list[int]],
    block_length: int = BLOCK_LENGTH,
    pad_id: int = PAD_ID,
    ignore_index: int = IGNORE_INDEX,
) -> dict:
    """Split each source window into inputs and one-shifted targets, plus masks.

    input_ids      = window positions 0:block_length
    targets        = window positions 1:block_length + 1, invalid -> ignore_index
    attention_mask = 1 at real input positions, 0 at padding
    loss_mask      = 1 only where both the input and its next token are real

    These targets are already shifted; a model API that shifts labels internally
    needs an adapter, not a second shift.

    Return a dict with exactly these keys, each a list of per-window lists
    (plain Python, shape `(B, T)`), so the fixture checks need no tensors:

        input_ids, targets, attention_mask, loss_mask
    """
    input_ids: list[list[int]] = []
    targets: list[list[int]] = []
    attention_mask: list[list[int]] = []
    loss_mask: list[list[int]] = []

    for window in source_windows:
        inputs = window[:block_length]
        next_tokens = window[1 : block_length + 1]
        input_ids.append(list(inputs))
        targets.append(
            [token if token != pad_id else ignore_index for token in next_tokens]
        )
        attention_mask.append([1 if token != pad_id else 0 for token in inputs])
        loss_mask.append(
            [
                1 if inputs[i] != pad_id and next_tokens[i] != pad_id else 0
                for i in range(len(inputs))
            ]
        )

    return {
        "input_ids": input_ids,
        "targets": targets,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
    }


def check_fixture_alignment() -> None:
    """Assert the hand-written fixtures from session4.md before using the corpus.

    Cover a full block, a short final block, a single-token input, a block ending
    at EOS, and an empty input. Assert shapes, targets, and both masks, and
    assert that the masks disagree where a real input has no real next token.
    """
    t = 3
    pad = PAD_ID
    eos = EOS_ID
    ignore = IGNORE_INDEX

    # 1. Full block: exactly T + 1 tokens, so no padding is needed.
    windows, info = build_source_windows([5, 6, 7, eos], t)
    assert windows == [[5, 6, 7, eos]], windows
    assert info["stream_length"] == 4
    assert info["windows"] == 1
    assert info["final_window_padded"] is False
    assert info["lone_trailing_token"] is None
    batch = make_lm_batch(windows, t)
    assert batch["input_ids"] == [[5, 6, 7]]
    assert batch["targets"] == [[6, 7, eos]]
    assert batch["attention_mask"] == [[1, 1, 1]]
    assert batch["loss_mask"] == [[1, 1, 1]]

    # 2. Short final block: right-padded; the padded position carries no loss.
    windows, info = build_source_windows([5, 6, eos], t)
    assert info["final_window_padded"] is True
    assert info["lone_trailing_token"] is None
    batch = make_lm_batch(windows, t)
    assert batch["input_ids"] == [[5, 6, eos]]
    assert batch["targets"] == [[6, eos, ignore]]
    assert batch["attention_mask"] == [[1, 1, 1]]
    assert batch["loss_mask"] == [[1, 1, 0]]

    # 3. Single token: no pair to score, so it is logged, not windowed.
    windows, info = build_source_windows([5], t)
    assert windows == []
    assert info["windows"] == 0
    assert info["final_window_padded"] is False
    assert info["lone_trailing_token"] == 5

    # 4. Block ending at EOS: the real 5 -> EOS pair stays trainable.
    windows, info = build_source_windows([5, eos], t)
    batch = make_lm_batch(windows, t)
    assert batch["input_ids"] == [[5, eos, pad]]
    assert batch["targets"] == [[eos, ignore, ignore]]
    assert batch["attention_mask"] == [[1, 1, 0]]
    assert batch["loss_mask"] == [[1, 0, 0]]
    # The two masks must disagree where a real input has no real next token.
    assert batch["loss_mask"] != batch["attention_mask"]

    # 5. Empty input is valid and yields nothing.
    windows, info = build_source_windows([], t)
    assert windows == []
    assert info["windows"] == 0
    assert info["lone_trailing_token"] is None
    batch = make_lm_batch(windows, t)
    assert batch["input_ids"] == []
    assert batch["targets"] == []
    assert batch["attention_mask"] == []
    assert batch["loss_mask"] == []

    print("check_fixture_alignment: 5 hand-written cases passed")


def main() -> None:
    """Wire the stages once the pieces above are implemented.

    Load the Session 3 training-only tokenizer artifact and the pinned split
    rows, build one stream per split, build examples, and print per-split counts,
    shapes, and one worked example. Keep splits separate throughout.
    """
    from wikitext_compare import record_rows, resolve_data_root, resolve_splits

    data_root = resolve_data_root()
    tokenizer_path = (
        data_root
        / "processed/week09_tokenization_text_data/session3_wikitext_trainonly"
        / "tokenizer.json"
    )
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Missing Session 3 training-only tokenizer artifact: {tokenizer_path}"
        )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    splits = resolve_splits(data_root)

    print(f"tokenizer: {tokenizer_path}")
    print(
        f"block_length={BLOCK_LENGTH} stride={STRIDE} "
        f"PAD={PAD_ID} EOS={EOS_ID} IGNORE={IGNORE_INDEX}"
    )

    example: dict | None = None
    example_split = ""

    for split_name, rows in splits.items():
        records = record_rows(rows)
        stream = build_split_stream(tokenizer, rows)
        windows, info = build_source_windows(stream)
        batch = make_lm_batch(windows)
        print(
            f"{split_name:10s} rows={len(rows):6d} records={len(records):6d} "
            f"stream={info['stream_length']:9d} windows={info['windows']:6d} "
            f"padded_final={info['final_window_padded']} "
            f"lone_trailing={info['lone_trailing_token']} "
            f"batch_keys={sorted(batch)}"
        )
        if example is None:
            example, example_split = batch, split_name

    assert example is not None
    print(f"one worked example, first window of the {example_split} split:")
    preview = tokenizer.decode(example["input_ids"][0])[:48]
    print(f"  text            {preview!r}")
    for key in ("input_ids", "targets", "attention_mask", "loss_mask"):
        print(f"  {key:15s} {example[key][0]}")


if __name__ == "__main__":
    main()
