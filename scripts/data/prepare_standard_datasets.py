#!/usr/bin/env python3
"""Prepare standard roadmap datasets outside the Git repository."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


DATASETS = {
    "mnist": {
        "kind": "torchvision",
        "class_name": "MNIST",
        "description": "MNIST handwritten digits via torchvision.",
    },
    "fashion_mnist": {
        "kind": "torchvision",
        "class_name": "FashionMNIST",
        "description": "Fashion-MNIST clothing images via torchvision.",
    },
    "cifar10": {
        "kind": "torchvision",
        "class_name": "CIFAR10",
        "description": "CIFAR-10 natural images via torchvision.",
    },
    "wikitext2": {
        "kind": "huggingface",
        "path": "Salesforce/wikitext",
        "name": "wikitext-2-raw-v1",
        "description": "WikiText-2 raw language-modeling corpus via Hugging Face datasets.",
    },
    "ag_news": {
        "kind": "huggingface",
        "path": "ag_news",
        "name": None,
        "description": "AG News text classification dataset via Hugging Face datasets.",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def resolve_data_root() -> Path:
    return Path(os.environ.get("AI_ROADMAP_DATA", "~/data/ai_research_roadmap")).expanduser()


def ensure_data_root(data_root: Path, allow_repo_data: bool) -> None:
    resolved_root = data_root.resolve()
    resolved_repo = repo_root().resolve()
    if is_relative_to(resolved_root, resolved_repo) and not allow_repo_data:
        raise SystemExit(
            f"Refusing to write dataset files inside the Git repo: {resolved_root}\n"
            "Set AI_ROADMAP_DATA to scratch/project storage, or pass --allow-repo-data intentionally."
        )
    for subdir in ("raw", "processed", "manifests", "tmp"):
        (data_root / subdir).mkdir(parents=True, exist_ok=True)


def prepare_torchvision(dataset_id: str, data_root: Path, dry_run: bool) -> dict:
    spec = DATASETS[dataset_id]
    raw_root = data_root / "raw" / "torchvision"
    if dry_run:
        return {
            "dataset": dataset_id,
            "kind": "torchvision",
            "target": str(raw_root),
            "action": "dry_run",
        }

    try:
        from torchvision import datasets as tv_datasets
    except ImportError as exc:
        raise SystemExit("torchvision is required for vision dataset preparation.") from exc

    dataset_class = getattr(tv_datasets, spec["class_name"])
    train = dataset_class(root=str(raw_root), train=True, download=True)
    test = dataset_class(root=str(raw_root), train=False, download=True)
    return {
        "dataset": dataset_id,
        "kind": "torchvision",
        "target": str(raw_root),
        "train_count": len(train),
        "test_count": len(test),
    }


def prepare_huggingface(dataset_id: str, data_root: Path, dry_run: bool) -> dict:
    spec = DATASETS[dataset_id]
    cache_dir = data_root / "raw" / "huggingface"
    if dry_run:
        return {
            "dataset": dataset_id,
            "kind": "huggingface",
            "target": str(cache_dir),
            "path": spec["path"],
            "name": spec["name"],
            "action": "dry_run",
        }

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("datasets is required for Hugging Face dataset preparation.") from exc

    if spec["name"]:
        dataset = load_dataset(spec["path"], spec["name"], cache_dir=str(cache_dir))
    else:
        dataset = load_dataset(spec["path"], cache_dir=str(cache_dir))

    split_counts = {split: len(rows) for split, rows in dataset.items()}
    return {
        "dataset": dataset_id,
        "kind": "huggingface",
        "target": str(cache_dir),
        "path": spec["path"],
        "name": spec["name"],
        "split_counts": split_counts,
    }


PREPARE_BY_KIND: dict[str, Callable[[str, Path, bool], dict]] = {
    "torchvision": prepare_torchvision,
    "huggingface": prepare_huggingface,
}


def write_manifest(data_root: Path, records: list[dict], dry_run: bool) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "created_at": timestamp,
        "data_root": str(data_root),
        "dry_run": dry_run,
        "records": records,
    }
    path = data_root / "manifests" / f"prepare_standard_datasets_{timestamp}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS),
        help="Dataset to prepare. Repeat this flag for multiple datasets.",
    )
    parser.add_argument("--list", action="store_true", help="List supported datasets and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be prepared without downloading.")
    parser.add_argument(
        "--allow-repo-data",
        action="store_true",
        help="Allow AI_ROADMAP_DATA to point inside the repo. Intended only for tiny local experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        for dataset_id, spec in sorted(DATASETS.items()):
            print(f"{dataset_id}: {spec['description']}")
        return

    if not args.dataset:
        raise SystemExit("Choose at least one --dataset, or use --list.")

    data_root = resolve_data_root()
    ensure_data_root(data_root, allow_repo_data=args.allow_repo_data)

    records = []
    for dataset_id in args.dataset:
        spec = DATASETS[dataset_id]
        prepare = PREPARE_BY_KIND[spec["kind"]]
        record = prepare(dataset_id, data_root, args.dry_run)
        records.append(record)
        print(json.dumps(record, sort_keys=True))

    manifest_path = write_manifest(data_root, records, dry_run=args.dry_run)
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
