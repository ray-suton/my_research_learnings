import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "README.md"


def read_file(path=DEFAULT_CORPUS_PATH):
    return Path(path).read_text(encoding="utf-8")


def split_examples(path=DEFAULT_CORPUS_PATH):
    txt = read_file(path)
    sentences = []
    for raw_line in txt.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", line)
        sentences.extend(part.strip() for part in parts if part.strip())    
    return sentences


def train_val_split(examples, val_fraction=0.2, seed=0):
    if not examples:
        raise ValueError("Cannot split an empty corpus")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    import random

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(round(len(shuffled) * val_fraction)))
    val_size = min(val_size, len(shuffled) - 1)
    return shuffled[val_size:], shuffled[:val_size]
