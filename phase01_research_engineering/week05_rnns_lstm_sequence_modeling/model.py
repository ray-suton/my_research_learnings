import torch
import torch.nn as nn


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_size=128, num_layers=1):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.embeddings(x)
        x, _ = self.lstm(x)
        logits = self.linear(x)
        return logits


class LSTMfull(CharLSTM):
    pass


def build_model(model_name, vocab_size, embedding_dim=64, hidden_size=128, num_layers=1):
    if model_name.lower() != "lstm":
        raise ValueError(f"Unknown model_name for Week 5: {model_name}")
    return CharLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
