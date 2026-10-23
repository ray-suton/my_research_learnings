import os 
from pathlib import Path 
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

CORPUS_PATH = Path(__file__).resolve().parents[2] / "README.md"
VOCAB_SIZE = 2048
PAD_TOKENS = "<pad>"
EOS_TOKENS = "<eos>"

PROBE_TEXTS = [
    "Hello, world!",
    "  leading and trailing  ",
    "tab\tseparated",
    "line1\nline2",
    "line1\r\nline2",
    "naïve — 日本語",
    "emoji 🙂🚀",
    "",
]

def read_training_text(path: Path) -> str:
    """Read UTF-8 text, preserving case, spaces, and newlines."""
    with path.open(encoding="utf-8", newline="") as file:
        text = file.read()
    if text == "":
        raise ValueError(f"Empty fitting corpus: {path}")
    return text


def train_tokenizer(training_text, vocab_size) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size = vocab_size,
        special_tokens = [PAD_TOKENS, EOS_TOKENS],
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(training_text, trainer = trainer)

    return tokenizer

def check_round_trip(tokenizer: Tokenizer, text: str) -> None:
    """Inspect pieces and IDs; assert exact ordinary-text reconstruction.

    Automatic special-token insertion is disabled. This check assumes the text
    does NOT contain literal "<pad>"/"<eos>": those strings are matched as the
    reserved IDs and are handled by check_reserved_literal_policy instead.
    """
    encoding = tokenizer.encode(text, add_special_tokens = False)
    print("pieces: ", encoding.tokens)
    print("ids:", encoding.ids)
    
    vocab_size = tokenizer.get_vocab_size()
    assert all(0 <= i < vocab_size for i in encoding.ids)

    decoded = tokenizer.decode(encoding.ids)
    assert decoded == text, f"round trip failed: {decoded!r} != {text!r}"

    re_encoded = tokenizer.encode(text, add_special_tokens = False)
    assert re_encoded.ids == encoding.ids, "encoding is not repeatable"


def check_reserved_literal_policy(tokenizer: Tokenizer) -> None:
    """Declare and check how literal "<pad>"/"<eos>" strings are handled.

    The bytes of an ordinary word and the reserved-token spelling are not the
    same thing: a literal "<pad>" in the text is matched as reserved ID 0, even
    with add_special_tokens=False. The default decode drops reserved tokens, so
    such text only round trips when decoded with skip_special_tokens=False.
    This is a separate policy and is not covered by check_round_trip.
    """
    sample = "a <pad> b <eos>"
    encoding = tokenizer.encode(sample, add_special_tokens=False)
    print("reserved-literal pieces:", encoding.tokens)
    print("reserved-literal ids:   ", encoding.ids)

    assert tokenizer.token_to_id(PAD_TOKENS) in encoding.ids, "literal <pad> not matched"
    assert tokenizer.token_to_id(EOS_TOKENS) in encoding.ids, "literal <eos> not matched"

    kept = tokenizer.decode(encoding.ids, skip_special_tokens=False)
    assert kept == sample, f"reserved-literal policy changed: {kept!r}"

    dropped = tokenizer.decode(encoding.ids)
    assert dropped != sample, "default decode unexpectedly kept reserved tokens"


def save_and_reload(tokenizer: Tokenizer, output_path: Path) -> Tokenizer:
    """Save the full pipeline outside Git and return the reloaded tokenizer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))

    reloaded = Tokenizer.from_file(str(output_path))

    assert reloaded.get_vocab() == tokenizer.get_vocab(), "vocabulary changed"
    assert reloaded.get_vocab_size() == tokenizer.get_vocab_size()
    for special in (PAD_TOKENS, EOS_TOKENS):
        assert reloaded.token_to_id(special) == tokenizer.token_to_id(special), (
            f"{special} id changed on reload"
        )

    for fixture in PROBE_TEXTS:
        original_ids = tokenizer.encode(fixture, add_special_tokens=False).ids
        reloaded_ids = reloaded.encode(fixture, add_special_tokens=False).ids
        assert reloaded_ids == original_ids, f"encoding changed for {fixture!r}"
        assert reloaded.decode(reloaded_ids) == fixture, f"decode changed for {fixture!r}"

    return reloaded

def main() -> None:
    training_text = read_training_text(CORPUS_PATH)
    tokenizer = train_tokenizer([training_text], VOCAB_SIZE)

    for probe in PROBE_TEXTS:
        check_round_trip(tokenizer, probe)
    check_reserved_literal_policy(tokenizer)

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

    for probe in PROBE_TEXTS:
        check_round_trip(reloaded, probe)
    check_reserved_literal_policy(reloaded)

if __name__ == "__main__":
    main()
