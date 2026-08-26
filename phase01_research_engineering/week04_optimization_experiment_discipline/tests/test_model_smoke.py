import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from model import build_model


def test_mini_resnet_forward_shape():
    model = build_model("mini_resnet")
    logits = model(torch.randn(4, 3, 32, 32))
    assert logits.shape == (4, 10)


def test_small_cnn_forward_shape():
    model = build_model("small_cnn")
    logits = model(torch.randn(4, 3, 32, 32))
    assert logits.shape == (4, 10)
