"""One-step manual LSTM block for understanding the gate math.

This file is intentionally small and artificial. It is not a replacement for
``torch.nn.LSTM``; it exposes the intermediate values that PyTorch normally
hides.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ManualLSTMParams:
    W_f: torch.Tensor
    b_f: torch.Tensor
    W_i: torch.Tensor
    b_i: torch.Tensor
    W_g: torch.Tensor
    b_g: torch.Tensor
    W_o: torch.Tensor
    b_o: torch.Tensor


def lstm_step(x_t, c_prev, h_prev, params):
    """Run one LSTM timestep and return every gate for inspection."""
    z_t = torch.cat((x_t, h_prev), dim=0)

    f_t = torch.sigmoid(z_t @ params.W_f + params.b_f)
    i_t = torch.sigmoid(z_t @ params.W_i + params.b_i)
    g_t = torch.tanh(z_t @ params.W_g + params.b_g)
    o_t = torch.sigmoid(z_t @ params.W_o + params.b_o)

    c_t = f_t * c_prev + i_t * g_t
    h_t = o_t * torch.tanh(c_t)

    return {
        "x_t": x_t,
        "h_prev": h_prev,
        "c_prev": c_prev,
        "z_t": z_t,
        "f_t": f_t,
        "i_t": i_t,
        "g_t": g_t,
        "o_t": o_t,
        "c_t": c_t,
        "h_t": h_t,
    }


def lstm_manual(x, c_prev, h_prev, W_g, W_o, b_g, b_o, W_i, b_i, W_f, b_f):
    """Compatibility wrapper around the learner's original parameter order."""
    params = ManualLSTMParams(
        W_f=W_f,
        b_f=b_f,
        W_i=W_i,
        b_i=b_i,
        W_g=W_g,
        b_g=b_g,
        W_o=W_o,
        b_o=b_o,
    )
    return lstm_step(x, c_prev, h_prev, params)


def make_tiny_example():
    """Create a deterministic two-unit LSTM example with easy shapes."""
    x_t = torch.tensor([1.0, -0.5])
    h_prev = torch.tensor([0.2, 0.1])
    c_prev = torch.tensor([0.7, -0.3])

    params = ManualLSTMParams(
        W_f=torch.tensor([[0.4, -0.2], [0.1, 0.3], [-0.5, 0.2], [0.2, 0.1]]),
        b_f=torch.tensor([0.0, 0.1]),
        W_i=torch.tensor([[0.3, 0.2], [-0.4, 0.1], [0.2, -0.3], [0.1, 0.4]]),
        b_i=torch.tensor([-0.1, 0.0]),
        W_g=torch.tensor([[0.2, -0.5], [0.3, 0.2], [0.4, 0.1], [-0.2, 0.3]]),
        b_g=torch.tensor([0.0, 0.05]),
        W_o=torch.tensor([[-0.3, 0.2], [0.5, -0.1], [0.1, 0.4], [0.2, 0.3]]),
        b_o=torch.tensor([0.05, -0.05]),
    )

    return x_t, c_prev, h_prev, params


def main():
    x_t, c_prev, h_prev, params = make_tiny_example()
    values = lstm_step(x_t=x_t, c_prev=c_prev, h_prev=h_prev, params=params)

    for name in ("x_t", "h_prev", "c_prev", "z_t", "f_t", "i_t", "g_t", "o_t", "c_t", "h_t"):
        value = values[name]
        print(f"{name:>6}: {value} shape={tuple(value.shape)}")

    expected_c_t = values["f_t"] * c_prev + values["i_t"] * values["g_t"]
    expected_h_t = values["o_t"] * torch.tanh(values["c_t"])
    assert torch.allclose(values["c_t"], expected_c_t)
    assert torch.allclose(values["h_t"], expected_h_t)
    print("manual LSTM block check passed")


if __name__ == "__main__":
    main()
