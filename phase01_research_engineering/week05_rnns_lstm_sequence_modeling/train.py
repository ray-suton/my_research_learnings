import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader, TensorDataset

from baseline import evaluate_bigram_baseline, fit_bigram_baseline, generate_bigram_sample
from config_utils import format_config, load_config, write_config
from corpus import split_examples, train_val_split
from model import build_model
from tokenizer import CharVocabulary, build_sequence_tensors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = PROJECT_ROOT / "results" / "runs"


def make_loader(X, y, batch_size, shuffle, seed, num_workers=0):
    dataset = TensorDataset(X, y)
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
    )


def masked_accuracy(logits, y, pad_id):
    preds = logits.argmax(dim=-1)
    mask = y != pad_id
    if mask.sum().item() == 0:
        return 0.0
    return (preds[mask] == y[mask]).float().mean().item()


def train_one_epoch(model, loader, loss_fn, optimizer, device, pad_id):
    model.to(device)
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_accuracy = 0.0
    total_batches = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
        y_flat = y.reshape(batch_size * seq_len)
        loss = loss_fn(logits_flat, y_flat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        non_pad_tokens = int((y != pad_id).sum().item())
        total_loss += loss.item() * non_pad_tokens
        total_tokens += non_pad_tokens
        total_accuracy += masked_accuracy(logits, y, pad_id)
        total_batches += 1

    return total_loss / total_tokens, total_accuracy / total_batches


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, pad_id):
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_accuracy = 0.0
    total_batches = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        batch_size, seq_len, vocab_size = logits.shape
        loss = loss_fn(logits.reshape(batch_size * seq_len, vocab_size), y.reshape(batch_size * seq_len))
        non_pad_tokens = int((y != pad_id).sum().item())
        total_loss += loss.item() * non_pad_tokens
        total_tokens += non_pad_tokens
        total_accuracy += masked_accuracy(logits, y, pad_id)
        total_batches += 1

    return total_loss / total_tokens, total_accuracy / total_batches


@torch.no_grad()
def generate_sample(model, vocabulary, prompt, device, max_new_chars=100, temperature=0.8, seed=0):
    model.to(device)
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    generated_ids = []
    for char in prompt:
        if char not in vocabulary.token_to_id:
            raise ValueError(f"Prompt contains character not in vocabulary: {char!r}")
        generated_ids.append(vocabulary.token_to_id[char])

    for _ in range(max_new_chars):
        x = torch.tensor([generated_ids], dtype=torch.long, device=device)
        logits = model(x)
        next_logits = logits[0, -1, :].detach().cpu() / temperature
        next_logits[vocabulary.pad_id] = float("-inf")
        probs = torch.softmax(next_logits, dim=0)
        next_id = torch.multinomial(probs, num_samples=1, generator=generator).item()
        if next_id == vocabulary.eos_id:
            break
        generated_ids.append(next_id)

    return vocabulary.decode_ids(generated_ids, stop_at_eos=True)


def make_run_dir(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_ROOT / f"{timestamp}_{config['run_name']}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def prepare_data(config):
    corpus_path = Path(config["corpus_path"])
    if not corpus_path.is_absolute():
        corpus_path = PROJECT_ROOT / corpus_path

    sentences = split_examples(corpus_path)
    train_sentences, val_sentences = train_val_split(
        sentences,
        val_fraction=config["val_fraction"],
        seed=config["seed"],
    )
    vocabulary = CharVocabulary.from_sentences(sentences)
    X_train, y_train, _, skipped_train = build_sequence_tensors(
        train_sentences,
        vocabulary=vocabulary,
        sequence_length=config["sequence_length"],
    )
    X_val, y_val, _, skipped_val = build_sequence_tensors(
        val_sentences,
        vocabulary=vocabulary,
        sequence_length=config["sequence_length"],
    )
    return {
        "corpus_path": corpus_path,
        "all_sentences": sentences,
        "train_sentences": train_sentences,
        "val_sentences": val_sentences,
        "vocabulary": vocabulary,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "skipped_train": skipped_train,
        "skipped_val": skipped_val,
    }


def write_run_record(run_dir, config, metrics, samples):
    write_config(run_dir / "config.yaml", config)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    sample_lines = ["# Generated Samples", ""]
    for prompt, sample in samples["lstm"].items():
        sample_lines.extend([f"## LSTM prompt {prompt!r}", "", sample, ""])
    for prompt, sample in samples["bigram"].items():
        sample_lines.extend([f"## Bigram prompt {prompt!r}", "", sample, ""])
    (run_dir / "samples.txt").write_text("\n".join(sample_lines), encoding="utf-8")

    notes = [
        f"# {config['run_name']}",
        "",
        "## Summary",
        "",
        f"- status: {metrics['status']}",
        f"- device: {metrics['device']}",
        f"- epochs: {metrics['epochs']}",
        f"- best validation loss: {metrics['best_val_loss']:.4f}",
        f"- final validation loss: {metrics['final_val_loss']:.4f}",
        f"- bigram validation loss: {metrics['bigram_val_loss']:.4f}",
        "",
        "## Notes",
        "",
        "- This is a tiny learning run, not a useful language model.",
    ]
    (run_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def train(config, record_run=True):
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    data = prepare_data(config)
    vocabulary = data["vocabulary"]
    train_loader = make_loader(
        data["X_train"],
        data["y_train"],
        batch_size=config["batch_size"],
        shuffle=True,
        seed=config["seed"],
        num_workers=config["num_workers"],
    )
    val_loader = make_loader(
        data["X_val"],
        data["y_val"],
        batch_size=config["batch_size"],
        shuffle=False,
        seed=config["seed"],
        num_workers=config["num_workers"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_name=config["model_name"],
        vocab_size=len(vocabulary),
        embedding_dim=config["embedding_dim"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocabulary.pad_id)
    optimizer = opt.Adam(model.parameters(), lr=config["learning_rate"])

    print(f"Running config:\n{format_config(config)}")
    start_time = time.time()
    history = []
    best_val_loss = float("inf")
    best_epoch = None
    for epoch in range(config["epochs"]):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            vocabulary.pad_id,
        )
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device, vocabulary.pad_id)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f}, "
            f"train_acc={train_accuracy:.4f}, val_loss={val_loss:.4f}, "
            f"val_acc={val_accuracy:.4f}"
        )

    bigram_probs = fit_bigram_baseline(data["train_sentences"], vocabulary)
    bigram_metrics = evaluate_bigram_baseline(data["val_sentences"], vocabulary, bigram_probs)
    samples = {
        "lstm": {
            prompt: generate_sample(
                model,
                vocabulary,
                prompt=prompt,
                device=device,
                max_new_chars=config["generate_num_chars"],
                temperature=config["temperature"],
                seed=config["seed"] + index,
            )
            for index, prompt in enumerate(config["prompts"])
        },
        "bigram": {
            prompt: generate_bigram_sample(
                prompt,
                vocabulary,
                bigram_probs,
                max_new_chars=config["generate_num_chars"],
                seed=config["seed"] + index,
            )
            for index, prompt in enumerate(config["prompts"])
        },
    }

    final_val_loss = history[-1]["val_loss"]
    metrics = {
        "status": "completed",
        "run_name": config["run_name"],
        "device": str(device),
        "corpus_path": str(data["corpus_path"]),
        "num_sentences": len(data["all_sentences"]),
        "num_train_sentences": len(data["train_sentences"]),
        "num_val_sentences": len(data["val_sentences"]),
        "skipped_train_examples": data["skipped_train"],
        "skipped_val_examples": data["skipped_val"],
        "vocab_size": len(vocabulary),
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "sequence_length": config["sequence_length"],
        "embedding_dim": config["embedding_dim"],
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
        "learning_rate": config["learning_rate"],
        "seed": config["seed"],
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_perplexity": math.exp(best_val_loss) if best_val_loss < 50 else float("inf"),
        "final_train_loss": history[-1]["train_loss"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_val_loss": final_val_loss,
        "final_val_accuracy": history[-1]["val_accuracy"],
        "final_val_perplexity": math.exp(final_val_loss) if final_val_loss < 50 else float("inf"),
        "bigram_val_loss": bigram_metrics["val_loss"],
        "bigram_val_perplexity": bigram_metrics["val_perplexity"],
        "bigram_val_transitions": bigram_metrics["transitions"],
        "runtime_seconds": round(time.time() - start_time, 3),
        "history": history,
        "samples": samples,
    }

    if record_run:
        run_dir = make_run_dir(config)
        metrics["run_dir"] = str(run_dir)
        write_run_record(run_dir, config, metrics, samples)
        print(f"Wrote run record to {run_dir}")

    print(json.dumps({key: metrics[key] for key in ("final_val_loss", "bigram_val_loss", "runtime_seconds")}, indent=2))
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
