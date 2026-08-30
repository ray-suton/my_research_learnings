import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import train as week5_train
from config_utils import load_config
from model import build_model
from tokenizer import CharVocabulary


CONFIG_DIR = PROJECT_DIR / "configs"


def test_config_loads():
    config = load_config(CONFIG_DIR / "lstm_readme_tiny.yaml")
    assert config["run_name"] == "lstm_readme_tiny"
    assert config["model_name"] == "lstm"


def test_model_forward_shape():
    model = build_model("lstm", vocab_size=11, embedding_dim=4, hidden_size=5, num_layers=1)
    logits = model(torch.randint(0, 11, (3, 7)))
    assert logits.shape == (3, 7, 11)


def test_tiny_training_loop_writes_run_record(tmp_path, monkeypatch):
    corpus_path = tmp_path / "tiny_corpus.txt"
    corpus_path.write_text("abab.\nbaba.\nabba.\nbaab.\n", encoding="utf-8")
    config = {
        "run_name": "pytest_tiny",
        "corpus_path": str(corpus_path),
        "model_name": "lstm",
        "epochs": 1,
        "batch_size": 2,
        "sequence_length": 8,
        "embedding_dim": 4,
        "hidden_size": 5,
        "num_layers": 1,
        "learning_rate": 0.01,
        "seed": 3,
        "num_workers": 0,
        "generate_num_chars": 8,
        "temperature": 0.8,
        "val_fraction": 0.25,
        "prompts": ["a"],
    }
    monkeypatch.setattr(week5_train, "RUNS_ROOT", tmp_path / "runs")

    metrics = week5_train.train(config, record_run=True)
    run_dirs = list((tmp_path / "runs").iterdir())

    assert metrics["status"] == "completed"
    assert metrics["epochs"] == 1
    assert metrics["bigram_val_loss"] > 0
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "config.yaml").exists()
    assert (run_dirs[0] / "metrics.json").exists()
    assert (run_dirs[0] / "samples.txt").exists()


def test_generation_rejects_unknown_prompt():
    vocab = CharVocabulary.from_sentences(["abc"])
    model = build_model("lstm", vocab_size=len(vocab), embedding_dim=4, hidden_size=5, num_layers=1)

    try:
        week5_train.generate_sample(model, vocab, "z", torch.device("cpu"))
    except ValueError as exc:
        assert "Prompt contains character not in vocabulary" in str(exc)
    else:
        raise AssertionError("unknown prompt should fail clearly")
