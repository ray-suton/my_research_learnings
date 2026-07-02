#!/usr/bin/env python3
"""Validate the roadmap dataset registry."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_TOP_LEVEL = {
    "schema_version",
    "data_root_env",
    "default_data_root",
    "storage_policy",
    "statuses",
    "datasets",
}

REQUIRED_DATASET_FIELDS = {
    "status",
    "phase",
    "weeks",
    "purpose",
    "source",
    "license",
    "storage",
    "prepare",
    "split",
    "redistribution",
    "leakage_notes",
}

REQUIRED_STORAGE_FIELDS = {"raw", "processed"}
REQUIRED_PREPARE_FIELDS = {"command", "script_supported"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="data_registry/datasets.yaml",
        help="Path to dataset registry YAML.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required to validate data_registry/datasets.yaml.") from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"{path} did not parse as a YAML mapping.")
    return data


def require_keys(mapping: dict, required: set[str], label: str) -> list[str]:
    missing = sorted(required.difference(mapping))
    return [f"{label} missing required key: {key}" for key in missing]


def validate_registry(data: dict) -> list[str]:
    errors: list[str] = []
    errors.extend(require_keys(data, REQUIRED_TOP_LEVEL, "registry"))
    if errors:
        return errors

    statuses = data["statuses"]
    if not isinstance(statuses, list) or not all(isinstance(item, str) for item in statuses):
        errors.append("registry statuses must be a list of strings")
        statuses = []
    allowed_statuses = set(statuses)

    datasets = data["datasets"]
    if not isinstance(datasets, dict) or not datasets:
        errors.append("registry datasets must be a non-empty mapping")
        return errors

    for dataset_id, spec in sorted(datasets.items()):
        label = f"dataset {dataset_id}"
        if not isinstance(spec, dict):
            errors.append(f"{label} must be a mapping")
            continue

        errors.extend(require_keys(spec, REQUIRED_DATASET_FIELDS, label))
        if "status" in spec and spec["status"] not in allowed_statuses:
            errors.append(f"{label} has unknown status: {spec['status']}")

        if "weeks" in spec and (not isinstance(spec["weeks"], list) or not spec["weeks"]):
            errors.append(f"{label} weeks must be a non-empty list")

        storage = spec.get("storage")
        if isinstance(storage, dict):
            errors.extend(require_keys(storage, REQUIRED_STORAGE_FIELDS, f"{label}.storage"))
        elif "storage" in spec:
            errors.append(f"{label}.storage must be a mapping")

        prepare = spec.get("prepare")
        if isinstance(prepare, dict):
            errors.extend(require_keys(prepare, REQUIRED_PREPARE_FIELDS, f"{label}.prepare"))
            if "script_supported" in prepare and not isinstance(prepare["script_supported"], bool):
                errors.append(f"{label}.prepare.script_supported must be boolean")
        elif "prepare" in spec:
            errors.append(f"{label}.prepare must be a mapping")

    return errors


def main() -> None:
    args = parse_args()
    path = Path(args.registry)
    data = load_yaml(path)
    errors = validate_registry(data)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print(f"validated {len(data['datasets'])} datasets from {path}")


if __name__ == "__main__":
    main()
