import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from manual_lstm_block import lstm_step, make_tiny_example


def test_manual_lstm_step_returns_all_gates_and_states():
    x_t, c_prev, h_prev, params = make_tiny_example()
    values = lstm_step(x_t, c_prev, h_prev, params)

    expected_keys = {"z_t", "f_t", "i_t", "g_t", "o_t", "c_t", "h_t"}
    assert expected_keys.issubset(values)
    assert values["z_t"].shape == (x_t.numel() + h_prev.numel(),)
    assert values["c_t"].shape == c_prev.shape
    assert values["h_t"].shape == h_prev.shape


def test_manual_lstm_state_update_matches_formula():
    x_t, c_prev, h_prev, params = make_tiny_example()
    values = lstm_step(x_t, c_prev, h_prev, params)

    expected_c_t = values["f_t"] * c_prev + values["i_t"] * values["g_t"]
    expected_h_t = values["o_t"] * torch.tanh(values["c_t"])

    assert torch.allclose(values["c_t"], expected_c_t)
    assert torch.allclose(values["h_t"], expected_h_t)
