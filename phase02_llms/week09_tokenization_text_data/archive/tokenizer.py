"""Session 2 learner scaffold: fit byte-level BPE on a tiny text file.

The root README is a practice input, not a held-out evaluation corpus.
Implement the four functions below in order. No tokenizer has been trained yet.
"""

import os
from pathlib import Path
from collections import Counter 
from tokenizers import Tokenizer


CORPUS_PATH = Path(__file__).resolve().parents[2] / "README.md"
VOCAB_SIZE = 2048


def read_training_text(path: Path) -> str:
    """Read UTF-8 text, preserving case, spaces, and newlines."""
    with path.open(encoding="utf-8", newline="") as file:
        return file.read()

def train_tokenizer(training_texts: list[str], vocab_size: int) -> Tokenizer:
    """Fit byte-level BPE using only these training records.

    Keep the full byte alphabet, distinct <pad>/<eos> tokens, and a matching
    byte-level decoder. Preserve ordinary text without normalization or an
    automatic prefix space. vocab_size is a target including reserved tokens.
    """
    sequences = [list(text) for text in training_texts]
    tokens = set()
    tokens = {token for sequence in sequences for token in sequence}
    while len(tokens) < vocab_size:
        pair_counts = Counter()
        for sequence in sequences:
            pair_counts.update(zip(sequence, sequence[1:]))
        if not pair_counts:
            break
        max_pair = None 
        max_count = 0
        new_token = None 
        for pair, count in pair_counts.items():
            if count > max_count:
                max_pair = pair 
                max_count = count 
        new_token = str(max_pair[0] + max_pair[1])
        tokens.add(new_token)
    
        for i, sequence in enumerate(sequences):
            new_list = []
            j = 0
            while j < len(sequence):
                if j + 1 < len(sequence) and (sequence[j], sequence[j+1]) == max_pair:
                    new_pair = sequence[j] + sequence[j+1]
                    new_list.append(new_pair)
                    j += 1
                else:
                    new_list.append(sequence[j])
                j += 1
            sequences[i] = new_list



def check_round_trip(tokenizer: Tokenizer, text: str) -> None:
    """Inspect pieces and IDs; assert exact ordinary-text reconstruction.

    Disable automatic special-token insertion. Literal reserved-token strings
    need a separate policy; this first check is for ordinary text.
    """
    raise NotImplementedError("Step 3: encode, inspect, decode, and compare")


def save_and_reload(tokenizer: Tokenizer, output_path: Path) -> Tokenizer:
    """Save the full pipeline outside Git and return the reloaded tokenizer.

    Create the output directory when implementing this step. Verify matching
    vocabulary/special IDs and encoding behavior on fixtures before accepting.
    """
    raise NotImplementedError("Step 4: save, reload, and verify behavior")


def main() -> None:
    training_text = read_training_text(CORPUS_PATH)
    tokenizer = train_tokenizer([training_text], VOCAB_SIZE)
    probe_text = "Hello, world!\n  Café 🙂"
    check_round_trip(tokenizer, probe_text)

    data_root = Path(
        os.environ.get(
            "AI_ROADMAP_DATA", "/home/rui.gao/data/ai_research_roadmap"
        )
    ).expanduser()
    output_path = (
        data_root
        / "processed/week09_tokenization_text_data/session2_readme"
        / "tokenizer.json"
    )
    reloaded = save_and_reload(tokenizer, output_path)
    check_round_trip(reloaded, probe_text)


if __name__ == "__main__":
    main()
