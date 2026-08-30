import math

import torch


def _token_sequences(sentences, vocabulary):
    return [vocabulary.encode_example(sentence) for sentence in sentences]


def fit_bigram_baseline(sentences, vocabulary, smoothing=1.0):
    vocab_size = len(vocabulary)
    counts = torch.zeros(vocab_size, vocab_size, dtype=torch.float64)
    valid_next_ids = [idx for idx in range(vocab_size) if idx != vocabulary.pad_id]

    for sequence in _token_sequences(sentences, vocabulary):
        for current_id, next_id in zip(sequence[:-1], sequence[1:]):
            counts[current_id, next_id] += 1.0

    counts[:, valid_next_ids] += smoothing
    probs = counts / counts.sum(dim=1, keepdim=True).clamp_min(1e-12)
    probs[:, vocabulary.pad_id] = 0.0
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return probs


def evaluate_bigram_baseline(sentences, vocabulary, probs):
    total_nll = 0.0
    total_pairs = 0
    for sequence in _token_sequences(sentences, vocabulary):
        for current_id, next_id in zip(sequence[:-1], sequence[1:]):
            prob = probs[current_id, next_id].clamp_min(1e-12).item()
            total_nll -= math.log(prob)
            total_pairs += 1

    if total_pairs == 0:
        raise ValueError("Cannot evaluate bigram baseline without token transitions")

    nll = total_nll / total_pairs
    return {
        "val_loss": nll,
        "val_perplexity": math.exp(nll) if nll < 50 else float("inf"),
        "transitions": total_pairs,
    }


def generate_bigram_sample(prompt, vocabulary, probs, max_new_chars=100, seed=0):
    generator = torch.Generator().manual_seed(seed)
    generated = list(prompt)

    for char in prompt:
        if char not in vocabulary.token_to_id:
            raise ValueError(f"Prompt contains character not in vocabulary: {char!r}")

    current_id = vocabulary.token_to_id[prompt[-1]]
    for _ in range(max_new_chars):
        next_id = torch.multinomial(probs[current_id].float(), num_samples=1, generator=generator).item()
        if next_id == vocabulary.eos_id:
            break
        generated.append(vocabulary.id_to_token[next_id])
        current_id = next_id

    return "".join(generated)
