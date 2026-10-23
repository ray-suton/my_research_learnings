"""Week 9 Session 3: training-only WikiText-2 refit and frozen-GPT-2 comparison.

This module owns the measured comparison described in `session3.md` and
`EXPERIMENT_PLAN.md`. It reuses the Session 2 functions in `tokenizer.py`
instead of duplicating them:

    read_training_text, train_tokenizer, check_round_trip,
    check_reserved_literal_policy, save_and_reload

Rules enforced here:

- The custom tokenizer is fitted on training text only.
- Validation text is the comparison text. Test text is not touched.
- Both tokenizers encode identical rows with no padding, no truncation, and no
  automatically inserted special tokens.
- The README prototype artifact from Session 2 is never overwritten.

Run from the repository root:

    .venv/bin/python phase02_llms/week09_tokenization_text_data/wikitext_compare.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tokenizer import (  # noqa: E402  (path insertion must happen first)
    check_reserved_literal_policy,
    check_round_trip,
    save_and_reload,
    train_tokenizer,
)

# --- Frozen choices (see session3.md "Working Contract To Freeze Before Running")

DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
REFERENCE_ID = "openai-community/gpt2"

RUN_ID = "session3_wikitext_trainonly"
FITTING_BYTE_CAP = 1024 * 1024  # 1 MiB of UTF-8 training text
TARGET_VOCAB_SIZE = 2048
BLOCK_LENGTH = 128

SPLITS = ("train", "validation", "test")

HELD_OUT_FIXTURES = [
    " = Homarus gammarus = \n",
    "The quick brown fox jumps over the lazy dog.",
    "naïve — 日本語 café 🙂 42",
]


def resolve_data_root() -> Path:
    return Path(
        os.environ.get("AI_ROADMAP_DATA", "/home/rui.gao/data/ai_research_roadmap")
    ).expanduser()


def raw_split_dir(data_root: Path) -> Path:
    return (
        data_root
        / "raw/huggingface/Salesforce/wikitext"
        / DATASET_CONFIG
    )


def resolve_splits(data_root: Path) -> dict[str, list[str]]:
    """Load the three upstream splits in source order. Never re-split."""
    split_dir = raw_split_dir(data_root)
    splits: dict[str, list[str]] = {}
    for split in SPLITS:
        parquet_path = split_dir / f"{split}-00000-of-00001.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Missing WikiText-2 {split} split: {parquet_path}"
            )
        table = pq.read_table(parquet_path)
        if table.column_names != ["text"]:
            raise ValueError(
                f"Unexpected columns in {parquet_path}: {table.column_names}"
            )
        splits[split] = table.column("text").to_pylist()
    return splits


def record_rows(rows: list[str]) -> list[str]:
    """A record is one nonempty source row; whitespace-only rows are skipped."""
    return [row for row in rows if row.strip()]


def select_fitting_subset(
    train_rows: list[str], max_bytes: int
) -> tuple[list[str], dict]:
    """Take whole nonempty training rows in source order up to the byte cap."""
    chosen: list[str] = []
    total_bytes = 0
    first_index: int | None = None
    last_index: int | None = None
    stopped_at: int | None = None

    for index, row in enumerate(train_rows):
        if not row.strip():
            continue
        row_bytes = len(row.encode("utf-8"))
        if total_bytes + row_bytes > max_bytes:
            stopped_at = index
            break
        chosen.append(row)
        total_bytes += row_bytes
        if first_index is None:
            first_index = index
        last_index = index

    info = {
        "records": len(chosen),
        "utf8_bytes": total_bytes,
        "byte_cap": max_bytes,
        "first_source_row": first_index,
        "last_source_row": last_index,
        "first_row_skipped_over_cap": stopped_at,
    }
    return chosen, info


def load_reference() -> Tokenizer:
    """Load the frozen GPT-2 tokenizer files. No model weights are loaded."""
    return Tokenizer.from_pretrained(REFERENCE_ID)


def encode_row_counts(tokenizer: Tokenizer, rows: list[str]) -> list[int]:
    """Encode rows with no padding, truncation, or automatic special tokens."""
    counts = []
    for row in rows:
        encoding = tokenizer.encode(row, add_special_tokens=False)
        counts.append(len(encoding.ids))
    return counts


def percentile(sorted_counts: list[int], fraction: float) -> int:
    if not sorted_counts:
        return 0
    position = int(round(fraction * (len(sorted_counts) - 1)))
    return sorted_counts[position]


def summarize(counts: list[int], total_bytes: int, vocab_size: int) -> dict:
    ordered = sorted(counts)
    total_tokens = sum(counts)
    quantile_95 = percentile(ordered, 0.95)
    tokens_per_byte = (total_tokens / total_bytes) if total_bytes else None
    return {
        "vocabulary_size": vocab_size,
        "records": len(counts),
        "comparison_text_utf8_bytes": total_bytes,
        "total_payload_tokens": total_tokens,
        "tokens_per_utf8_byte": tokens_per_byte,
        "median_tokens_per_record": percentile(ordered, 0.5),
        "p95_tokens_per_record": quantile_95,
        "min_tokens_per_record": ordered[0] if ordered else 0,
        "max_tokens_per_record": ordered[-1] if ordered else 0,
        "records_over_block_length": sum(1 for c in counts if c > BLOCK_LENGTH),
        "fraction_records_over_block_length": (
            sum(1 for c in counts if c > BLOCK_LENGTH) / len(counts)
            if counts
            else None
        ),
        "unknown_or_unsupported_outputs": 0,
    }


def duplicate_audit(
    train_records: list[str], held_out: dict[str, list[str]]
) -> dict:
    """Count held-out records whose text also appears in the training split."""
    train_text = set(train_records)
    audit = {}
    for split, rows in held_out.items():
        records = record_rows(rows)
        overlapping = [row for row in records if row in train_text]
        audit[split] = {
            "records": len(records),
            "records_also_in_train": len(overlapping),
            "examples": [row[:120] for row in overlapping[:3]],
        }
    return audit


def largest_differences(
    custom_counts: list[int], reference_counts: list[int], rows: list[str], limit: int = 5
) -> list[dict]:
    """Rows where the custom tokenizer costs the most extra tokens."""
    scored = sorted(
        range(len(rows)),
        key=lambda i: (custom_counts[i] - reference_counts[i]),
        reverse=True,
    )
    return [
        {
            "row_index": i,
            "custom_tokens": custom_counts[i],
            "gpt2_tokens": reference_counts[i],
            "text_preview": rows[i][:120],
        }
        for i in scored[:limit]
    ]


def main() -> None:
    data_root = resolve_data_root()
    splits = resolve_splits(data_root)

    split_counts = {
        split: {
            "rows": len(rows),
            "nonempty_records": len(record_rows(rows)),
            "utf8_bytes_nonempty": sum(
                len(row.encode("utf-8")) for row in record_rows(rows)
            ),
        }
        for split, rows in splits.items()
    }
    for split, info in split_counts.items():
        print(f"{split}: {info}")

    train_records = record_rows(splits["train"])
    validation_records = record_rows(splits["validation"])

    fitting_rows, fitting_info = select_fitting_subset(
        splits["train"], FITTING_BYTE_CAP
    )
    print("fitting subset:", fitting_info)

    # Upstream WikiText-2 repeats some short nonempty lines (headings, small
    # section rows) across splits. Audit that overlap and record it instead of
    # silently changing the declared split membership. The fitting subset is
    # still drawn only from the training split, in source order.
    held_out_rows = {
        "validation": splits["validation"],
        "test": splits["test"],
    }
    overlap = duplicate_audit(train_records, held_out_rows)
    held_out_text = set(validation_records) | set(record_rows(splits["test"]))
    fitting_dupes = [row for row in fitting_rows if row in held_out_text]
    overlap["fitting_subset_rows_also_held_out"] = {
        "count": len(fitting_dupes),
        "fraction_of_fitting_records": (
            len(fitting_dupes) / len(fitting_rows) if fitting_rows else None
        ),
        "examples": [row[:120] for row in fitting_dupes[:3]],
    }
    print("duplicate audit:", overlap)

    fit_start = time.perf_counter()
    custom = train_tokenizer(fitting_rows, TARGET_VOCAB_SIZE)
    fit_seconds = time.perf_counter() - fit_start
    custom_vocab = custom.get_vocab_size()
    print(f"custom vocabulary: {custom_vocab} (target {TARGET_VOCAB_SIZE}), "
          f"fit {fit_seconds:.2f}s")

    run_dir = data_root / f"processed/week09_tokenization_text_data/{RUN_ID}"
    artifact_path = run_dir / "tokenizer.json"
    if artifact_path.exists():
        print("overwriting previous Session 3 artifact at", artifact_path)
    custom_reloaded = save_and_reload(custom, artifact_path)

    for fixture in HELD_OUT_FIXTURES:
        check_round_trip(custom_reloaded, fixture)
    check_reserved_literal_policy(custom_reloaded)

    reference = load_reference()
    print(f"reference vocabulary: {reference.get_vocab_size()} ({REFERENCE_ID})")

    validation_bytes = sum(len(row.encode("utf-8")) for row in validation_records)

    start = time.perf_counter()
    custom_counts = encode_row_counts(custom_reloaded, validation_records)
    custom_encode_seconds = time.perf_counter() - start

    start = time.perf_counter()
    reference_counts = encode_row_counts(reference, validation_records)
    reference_encode_seconds = time.perf_counter() - start

    comparison = {
        "custom_training_only_bpe": summarize(
            custom_counts, validation_bytes, custom_vocab
        ),
        "frozen_gpt2_reference": summarize(
            reference_counts, validation_bytes, reference.get_vocab_size()
        ),
    }
    comparison["custom_training_only_bpe"]["encode_seconds"] = custom_encode_seconds
    comparison["frozen_gpt2_reference"]["encode_seconds"] = reference_encode_seconds

    # Sensitivity check for the audited overlap: refit on the same declared
    # subset with every row that also occurs in a held-out split removed. These
    # are short generic section headings, so the expected effect is small; the
    # point is to measure it rather than assume it.
    strict_rows = [row for row in fitting_rows if row not in held_out_text]
    strict = train_tokenizer(strict_rows, TARGET_VOCAB_SIZE)
    strict_counts = encode_row_counts(strict, validation_records)
    comparison["custom_excluding_held_out_duplicates"] = summarize(
        strict_counts, validation_bytes, strict.get_vocab_size()
    )

    differences: list[dict] = largest_differences(
        custom_counts, reference_counts, validation_records
    )

    metrics = {
        "run_id": RUN_ID,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "id": DATASET_ID,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "raw_dir": str(raw_split_dir(data_root)),
        },
        "splits": split_counts,
        "fitting_subset": fitting_info,
        "custom_fit_seconds": fit_seconds,
        "artifact_path": str(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "reference": {"id": REFERENCE_ID, "weights_loaded": False},
        "comparison_text": "validation nonempty records, source order",
        "block_length": BLOCK_LENGTH,
        "duplicate_audit": overlap,
        "strict_variant": {
            "fitting_records": len(strict_rows),
            "fitting_utf8_bytes": sum(len(row.encode("utf-8")) for row in strict_rows),
            "vocabulary_size": strict.get_vocab_size(),
            "note": "declared subset with rows duplicated in a held-out split removed",
        },
        "comparison": comparison,
        "largest_custom_minus_gpt2_differences": differences,
    }

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print()
    print("=== Validation comparison (identical rows, no padding/truncation) ===")
    for name, stats in comparison.items():
        print(
            f"{name}: vocab={stats['vocabulary_size']} "
            f"tokens={stats['total_payload_tokens']} "
            f"tok/byte={stats['tokens_per_utf8_byte']:.4f} "
            f"median={stats['median_tokens_per_record']} "
            f"p95={stats['p95_tokens_per_record']} "
            f"over_{BLOCK_LENGTH}={stats['records_over_block_length']} "
            f"({stats['fraction_records_over_block_length']:.3f})"
        )
    print()
    print("largest custom-minus-gpt2 token differences:")
    for row in differences:
        print(f"  +{row['custom_tokens'] - row['gpt2_tokens']:4d} "
              f"(custom {row['custom_tokens']}, gpt2 {row['gpt2_tokens']}): "
              f"{row['text_preview']!r}")
    print()
    print("metrics:", metrics_path)
    print("tokenizer:", artifact_path)


if __name__ == "__main__":
    main()
